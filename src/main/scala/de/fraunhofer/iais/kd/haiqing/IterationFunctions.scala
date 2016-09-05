package de.fraunhofer.iais.kd.haiqing


import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import breeze.linalg.{*, CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, axpy => Baxpy, max => Bmax, min => Bmin, sum => Bsum}
import breeze.numerics.{ceil, abs => Babs, exp => Bexp, log => Blog, sigmoid => Bsigmoid}
import breeze.stats.distributions.Multinomial
import de.fraunhofer.iais.kd.util.AliasMethod

/**
  * Created by hwang on 04.03.16.
  */
//val iterFct = new IterationFunctions(window, vectorSize, maxNumSenses, numNegative, vocabSize,learningRate, null,
//senseCount, expTable, numberOfSensesPerWord, smoothedFrequencyLookupTable, syn0, syn1)

/**
  * all constants for the model training / sense assignment
  *
  * @param window
  * @param vectorSize
  * @param maxNumSenses
  * @param numNegative
  * @param learningRate
  * @param ENCODE
  * @param gamma
  * @param oneSense            use only one sense.
  * @param softMax             use softmax activation, else sigmoid
  * @param modelPathOneSense   directory to store one-sense model
  * @param modelPathMultiSense directory to store multi-sense model
  * @param modelSaveIter       store model after this number of iterations
  * @param modelValidateIter   validate model after this number of iterations
  * @param maxEmbNorm          limit the length of emebbings to  this number
  * @param senseProbThresh     re-initialize embedding if senseProbability is lower
  * @param printLv             level for training output >=0
  * @param weightDecay         factor for reducing weights during a whole training epoch ( e.g. 0.1)
  * @param syn1OneSense        syn1 has only one sense
  */
class ModelConst(val window: Int, val vectorSize: Int, val maxNumSenses: Int,
                 val numNegative: Int, val learningRate: Float, val ENCODE: Int,
                 val gamma: Float, val oneSense: Boolean, val softMax: Boolean,
                 val evaluationWordPath: String,
                 val modelPathOneSense: String, val modelPathMultiSense: String, val modelSaveIter: Int,
                 val modelValidateIter: Int, val maxEmbNorm: Float, val senseProbThresh: Float, val printLv: Int,
                 val weightDecay: Float, val syn1OneSense: Boolean)
  extends Serializable {

  println("---- ModelConst Parameters: \n oneSense=" + oneSense + "\n softMax=" + softMax + "\n modelSaveIter=" +
    modelSaveIter
    + "\n modelValidateIter=" + modelValidateIter + "\n maxEmbNorm=" + maxEmbNorm
    + "\n senseProbThresh=" + senseProbThresh
    + "\n printLv=" + printLv + "\n weightDecay=" + weightDecay+ "\n syn1OneSense="+syn1OneSense)

  val MAX_EXP = 20
  require(maxNumSenses <= ENCODE, "NOT maxNumSenses<=ENCODE")
  require(window > 0, "NOT window>0")
  var vocabSize: Int = -1
  var numberOfSensesPerWord: Array[Int] = null
  //var smoothedFrequencyLookupTable: Array[Int] = null
  var multinomFreqDistr: AliasMethod = null
  // generator for word frequencies
  private val EXP_TABLE_SIZE = 100000
  // for small values the loss is erratic
  val expTable = createExpTable()
  val maxAdjusting = 10
  val wordCountFactor = 0.8
  val senseInitStandardDev = 0.01f // perturb read seses with this standard deviation


  @inline final def dotProd(x: Array[Float], y: Array[Float]): Float = {
    blas.sdot(x.length, x, 1, y, 1)
    //    var sm = 0.0f
    //    require(x.length == y.length)
    //    for (j <- 0 until x.length)
    //      sm += x(j) * y(j)
    //    require(sm==sm1)
    //    sm
  }


  def initSynRand(rand: util.Random, sd: Float = 1.0f): Array[Array[Array[Float]]] = {
    val syn = new Array[Array[Array[Float]]](vocabSize)
    for (word <- 0 until vocabSize) {
      syn(word) = new Array[Array[Float]](numberOfSensesPerWord(word))
      for (sense <- 0 until numberOfSensesPerWord(word)) {
        syn(word)(sense) = Array.fill[Float](vectorSize)((rand.nextFloat() - 0.5f) * sd / vectorSize)
      }
    }
    syn
  }

  def syn02bdv(syn: Array[Array[Array[Float]]], parm: ArrayBuffer[Double]) = {
    for (word <- 0 until syn.length) {
      for (sense <- 0 until syn(word).length) {
        for (i <- 0 until syn(word)(sense).length) {
          parm += syn(word)(sense)(i)
        }
      }
    }
  }


  def bdv2names(): Array[String] = {
    val res = new ArrayBuffer[String]()
    for (sy <- Array("syn0", "syn1")) {
      for (word <- 0 until vocabSize) {
        for (sense <- 0 until numberOfSensesPerWord(word)) {
          for (i <- 0 until vectorSize) {
            res += sy + "(w" + word + ",s" + sense + "," + i + ")"
          }
        }
      }
    }
    res.toArray
  }


  def bdv2syn0(parm: BDV[Double], ioff: Int, syn: Array[Array[Array[Float]]]): Int = {
    var iof = ioff
    for (word <- 0 until vocabSize) {
      require(syn(word).length == numberOfSensesPerWord(word), "syn0(word).length")
      for (sense <- 0 until numberOfSensesPerWord(word)) {
        require(syn(word)(sense).length == vectorSize, "syn0(word)(sense).length")
        for (i <- 0 until vectorSize) {
          syn(word)(sense)(i) = parm(iof).toFloat
          iof += 1
        }
      }
    }
    iof
  }

  def syn2bdv(syn0: Array[Array[Array[Float]]], syn1: Array[Array[Array[Float]]]): BDV[Double] = {
    val parm = new ArrayBuffer[Double]()
    syn02bdv(syn0, parm)
    syn02bdv(syn1, parm)
    val res = BDV(parm.toArray)
    res
  }


  def bdv2syn(parm: BDV[Double]): (Array[Array[Array[Float]]], Array[Array[Array[Float]]]) = {
    val nparam = getTotalNumberOfParams()
    require(nparam > 0, "NOT nparam>0")
    val sy0 = initSynZero()
    val sy1 = initSynZero()

    var ioff = 0
    ioff = bdv2syn0(parm, ioff, sy0)
    ioff = bdv2syn0(parm, ioff, sy1)
    require(ioff == 2 * nparam, "NOT ioff==nparam")
    (sy0, sy1)
  }


  //initialize syn0, syn1 to 0
  def initSynZero(): Array[Array[Array[Float]]] = {
    ModelConst.initSynZero(numberOfSensesPerWord, vectorSize)
  }

  //initialize syn0, syn1 to 0
  def synEqual(s0: Array[Array[Array[Float]]], s1: Array[Array[Array[Float]]]) {
    require(s0.length == s1.length, "NOT s0.length==s1.length")
    for (iw <- 0 until vocabSize) {
      require(s0(iw).length == s1(iw).length, "NOT s0(iw).length==s1(iw).length")
      for (sense <- 0 until numberOfSensesPerWord(iw)) {
        require(s0(iw)(sense).length == s1(iw)(sense).length, "NOT s0(iw)(sense).length==s1(iw)(sense).length")
        for (j <- 0 until vectorSize)
          require(s0(iw)(sense)(j) == s0(iw)(sense)(j), "NOT iw" + iw + " sense" + sense + "j" + j)
      }
    }
  }

  /**
    * limit the length of embeddings to avoid overflow
    *
    * @param syn
    * @param maxlen
    */

  def limitEmbeddingLength(syn: Array[Array[Array[Float]]], maxlen: Double = 4.0): Int = {
    var numLimit = 0
    var check = true
    for (word <- 0 until syn.length) {
      for (sense <- 0 until syn(word).length) {
        val len = dotProd(syn(word)(sense), syn(word)(sense))
        if (len > maxlen) {
          numLimit += 1
          val fac = math.sqrt(maxlen * 0.99 / len)
          for (j <- 0 until syn(word)(sense).length)
            syn(word)(sense)(j) *= fac.toFloat
          if (check) {
            val len1 = dotProd(syn(word)(sense), syn(word)(sense))
            require(len1 < maxlen, "NOT len1<maxlen")
            check = false
          }
        }
      }
    }
    numLimit
  }

  //initialize syn0, syn1 to 0
  def synClone(syn: Array[Array[Array[Float]]]): Array[Array[Array[Float]]] = {
    val synNew = new Array[Array[Array[Float]]](syn.length)
    for (word <- 0 until syn.length) {
      synNew(word) = new Array[Array[Float]](syn(word).length)
      for (sense <- 0 until syn(word).length) {
        synNew(word)(sense) = new Array[Float](vectorSize)
        for (j <- 0 until vectorSize)
          synNew(word)(sense)(j) = syn(word)(sense)(j)
      }
    }
    synNew
  }


  def getEXP_TABLE_SIZE = EXP_TABLE_SIZE

  def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      // i=0 -> exp(-MAX_EXP)=exp(-6)  and i=EXP_TABLE_SIZE-1  -> exp(MAX_EXP)=exp(6)
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }

  def setDictInfo(vocabSize: Int, numberOfSensesPerWord: Array[Int], multinomFreqDistr: AliasMethod): Unit = {
    require(this.numberOfSensesPerWord == null, "NOT this.numberOfSensesPerWord==null ")
    require(this.multinomFreqDistr == null, "NOT this.multinomFreqDistr==null ")
    this.vocabSize = vocabSize
    this.numberOfSensesPerWord = numberOfSensesPerWord
    this.multinomFreqDistr = multinomFreqDistr
    require(numberOfSensesPerWord.length == vocabSize, "NOT numberOfSensesPerWord.length==vocabSize")
  }

  def getTotalNumberOfParams(): Int = {
    var nw = 0
    for (iw <- 0 until numberOfSensesPerWord.length)
      nw += numberOfSensesPerWord(iw) * vectorSize
    nw
  }

  def unstack(vs: Array[Float], ioff: Int, div: Float = 1.0f): Array[Array[Array[Float]]] = {
    var ioff0 = ioff
    val nw = numberOfSensesPerWord.length
    val syn0 = new Array[Array[Array[Float]]](nw)
    for (iw <- 0 until numberOfSensesPerWord.length) {
      val ns = numberOfSensesPerWord(iw)
      syn0(iw) = new Array[Array[Float]](ns)
      for (is <- 0 until ns) {
        syn0(iw)(is) = new Array[Float](vectorSize)
        for (j <- 0 until vectorSize) {
          syn0(iw)(is)(j) = vs(ioff0) / div
          ioff0 += 1
        }
      }
    }
    require(ioff0 == getTotalNumberOfParams(), "NOT ioff0==getTotalNumberOfParams()")
    syn0
  }

  def stackSynIntoArray(syn: Array[Array[Array[Float]]]): Array[Float] = {
    var nw = getTotalNumberOfParams()
    val res = new Array[Float](nw)

    var ioff = 0
    for (iw <- 0 until numberOfSensesPerWord.length) {
      for (is <- 0 until numberOfSensesPerWord(iw)) {
        for (j <- 0 until vectorSize) {
          res(ioff) = syn(iw)(is)(j)
          ioff += 1
        }
      }
    }
    require(ioff == nw)
    res
  }

  /**
    * generate numNegative negative examples different from posWs
    *
    * @param posWs current positive word-sense
    * @return
    */
  def getNEG(posWs: Int, rand: Random): Array[Int] = {
    val negSamples = new Array[Int](numNegative)
    for (i <- 0 until numNegative) {
      var negWord = multinomFreqDistr.next()
      while (negWord == wrd(posWs)) {
        negWord = multinomFreqDistr.next()
      }
      //add sense information (assign sense randomly)
      // sense: value in 0 . .. numberOfSensesPerWord(negWord)
      val randSense =
        if (numberOfSensesPerWord(negWord) > 1 && !syn1OneSense) rand.nextInt(numberOfSensesPerWord(negWord))
        else 0
      negSamples(i) = negWord * ENCODE + randSense
      require(0 <= wrd(negSamples(i)) && wrd(negSamples(i)) < vocabSize)
      require(0 <= sns(negSamples(i)) && sns(negSamples(i)) < numberOfSensesPerWord(negWord))

    }
    negSamples
  }


  /**
    * generate negative word-senses for a whole sentence
    *
    * @param sentence current positive words of sentence
    * @return
    */
  def generateSentenceNEG(sentence: Array[Int], rand: Random): Array[Array[Int]] = {
    val negs = new Array[Array[Int]](sentence.length)
    for (pos <- 0 until sentence.length) {
      negs(pos) = getNEG(sentence(pos), rand)
    }
    negs
  }


  def initSenseCounts(numberOfSensesPerWord: Array[Int]): Array[Array[Int]] = {
    val senseCount = new Array[Array[Int]](numberOfSensesPerWord.length) // an array of Int for the senses of each word
    for (w <- 0 until numberOfSensesPerWord.length) {
      senseCount(w) = new Array[Int](numberOfSensesPerWord(w)) // length of array = number of senses
    }
    senseCount
  }

  /**
    * access functions
    *
    * @param ws
    * @return
    */
  @inline final def wrd(ws: Int): Int = {
    ws / ENCODE
  }

  @inline final def sns(ws: Int): Int = {
    ws % ENCODE
  }

  @inline final def emb(syn: Array[Array[Array[Float]]], ws: Int): Array[Float] = {
    syn(ws / ENCODE)(ws % ENCODE)
  }

  @inline final def embSet(syn: Array[Array[Array[Float]]], ws: Int, iemb: Int, newValue: Float): Unit = {
    syn(ws / ENCODE)(ws % ENCODE)(iemb) = newValue
  }

  @inline final def embInc(syn: Array[Array[Array[Float]]], ws: Int, iemb: Int, newValue: Float): Unit = {
    syn(ws / ENCODE)(ws % ENCODE)(iemb) += newValue
  }

}

object ModelConst {

  //initialize syn0, syn1 to 0
  def initSynZero(numberOfSensesPerWord: Array[Int], vectorSize: Int): Array[Array[Array[Float]]] = {
    val syn = new Array[Array[Array[Float]]](numberOfSensesPerWord.length)
    for (word <- 0 until numberOfSensesPerWord.length) {
      syn(word) = new Array[Array[Float]](numberOfSensesPerWord(word))
      for (sense <- 0 until numberOfSensesPerWord(word)) {
        syn(word)(sense) = new Array[Float](vectorSize)
      }
    }
    syn
  }
}

class ModelUpdater(val m: ModelConst, seed: Long, sy0: Array[Array[Array[Float]]], sy1: Array[Array[Array[Float]]]) {
  //private var sentence: Array[Int] = null
  //private var sentenceNEG: Array[Array[Int]] = null
  val rand: Random = new util.Random(seed)
  val senseCountAdd = new Array[Array[Int]](m.vocabSize) // an array of Int for the senses of each word: to be added
  for (w <- 0 until m.vocabSize) {
    senseCountAdd(w) = new Array[Int](m.numberOfSensesPerWord(w))
  }
  // need to copy as otherwise in local mode all syn parameters are the same in all kernels
  val syn0: Array[Array[Array[Float]]] = m.synClone(sy0)
  val syn1: Array[Array[Array[Float]]] = m.synClone(sy1)
  //var getLossDrv:(Int, Int,Array[Int],Array[Array[Int]], Boolean) => Unit


  //set sentence
  //  def setSentence(sentence: Array[Int]): Unit = {
  //    this.sentence = sentence
  //  }
  //
  //  //set sentence negative samplings
  //  def setSentenceNEG(sentenceNEG: Array[Array[Int]]): Unit = {
  //    this.sentenceNEG = sentenceNEG
  //  }

  /**
    * select the best sense of the word for each position in the sentence.
    * the negative examples are aplways the same and yield an additive component of the loss-> may be omitted
    *
    * @param sentence
    * @param sentenceNEG
    * @return
    */
  def adjustSentence(sentence: Array[Int], sentenceNEG: Array[Array[Int]]): (Boolean) = {
    var adjusted = false
    for (pos <- 0 until sentence.size) {
      val word = sentence(pos) / m.ENCODE
      if (m.numberOfSensesPerWord(word) > 1) {
        var bestSense = -1 //there is no best sense
        var minLoss = Double.MaxValue
        for (sense <- 0 until m.numberOfSensesPerWord(word)) {
          val w = word * m.ENCODE + sense
          val loss = getLoss(w, pos, sentence, sentenceNEG)._1
          if (loss < minLoss) {
            minLoss = loss
            bestSense = sense
          }
        }
        val newWs = word * m.ENCODE + bestSense
        if (newWs != sentence(pos)) {
          adjusted = true
          sentence(pos) = newWs
        }
      }
    }
    (adjusted)
  }

  /**
    * compute the loss of a token-sense w at position pos in sentence
    * log probability of using center word to predict surrounding words
    * loss = - score, skip-gram score from whole sentence
    *
    * @param sentence    the current sentence words
    * @param sentenceNEG negative examples. may be null -> then omit negative examples
    *                    if negative samples are identical their additive contribution is always the same
    *                    this applies for selection of best sense and for the validation set
    * @return
    */

  def sentenceLoss(sentence: Array[Int], sentenceNEG: Array[Array[Int]]): (Double, Int) = {
    var loss = 0.0
    var lossNum = 0
    for (pos <- 0 to sentence.size - 1) {
      val tmp = getLoss(sentence(pos), pos, sentence, sentenceNEG)
      loss += tmp._1
      lossNum += tmp._2
    }
    (loss, lossNum)
  }

  /**
    * compute the loss of a token-sense w at position pos in sentence
    * log probability of using center word to predict surrounding words
    *
    * @param inpWs       input word-sense at pos in sentence
    * @param pos         positiuon in sentence
    * @param sentence    the current sentence words
    * @param sentenceNEG negative examples. may be null -> then omit negative examples
    *                    if negative samples are identical their additive contribution is always the same
    *                    this applies for selection of best sense and for the validation set
    * @return
    */
  def getLoss(inpWs: Int, pos: Int, sentence: Array[Int], sentenceNEG: Array[Array[Int]]): (Double, Int) = {
    var loss = 0.0
    var lossNum = 0
    if (m.softMax) {
      /*--------------------- SOFTMAX----------------------------*/
      for (p <- pos - m.window to pos + m.window) {
        if (p != pos && p >= 0 && p < sentence.size) {
          val x = m.dotProd(emb(syn0, inpWs), emb(syn1, sentence(p)))
          val negs: Array[Int] = sentenceNEG(p)
          val y = new Array[Double](negs.length)
          for (j <- 0 until negs.length) {
            y(j) = m.dotProd(emb(syn0, inpWs), emb(syn1, negs(j)))
          }
          val (logSmxX, derivX, derivY) = logSoftmax(x, y, false)
          //println("outWs=" + sentence(p) + " inpWs=" + inpWs + " softmx=" + logSmxX)
          loss += -logSmxX
          lossNum += 1
        }
      }
    } else {
      /*--------------------- SIGMOID----------------------------*/
      for (p <- pos - m.window to pos + m.window) {
        if (p != pos && p >= 0 && p < sentence.size) {
          val outWs = if (m.syn1OneSense) wrdSns0(sentence(p)) else sentence(p)
          loss +=
            -math.log(actFunSigmoidApprox(emb(syn0, inpWs), emb(syn1, outWs)))
          lossNum += 1
          if (sentenceNEG != null) {
            val NEG = sentenceNEG(p)
            for (nWs <- NEG) {
              val negWs = if (m.syn1OneSense) wrdSns0(nWs) else nWs
              loss += -math.log(1 - actFunSigmoidApprox(emb(syn0, inpWs), emb(syn1, negWs)))
              lossNum += 1
            }
          }
        }
      }
    }
    (loss, lossNum)
  }

  @inline final def wrd(ws: Int): Int = {
    ws / m.ENCODE
  }

  /**
    * generate wordsense with same word and sense 0
    *
    * @param ws
    * @return
    */
  @inline final def wrdSns0(ws: Int): Int = {
    (ws / m.ENCODE) * m.ENCODE
  }

  @inline final def sns(ws: Int): Int = {
    ws % m.ENCODE
  }

  @inline final def emb(syn: Array[Array[Array[Float]]], ws: Int): Array[Float] = {
    syn(ws / m.ENCODE)(ws % m.ENCODE)
  }

  @inline final def embSet(syn: Array[Array[Array[Float]]], ws: Int, iemb: Int, newValue: Float): Unit = {
    syn(ws / m.ENCODE)(ws % m.ENCODE)(iemb) = newValue
  }

  @inline final def embInc(syn: Array[Array[Array[Float]]], ws: Int, iemb: Int, newValue: Float): Unit = {
    require(0 <= wrd(ws) && wrd(ws) < m.vocabSize, "ws" + ws)
    require(0 <= sns(ws) && sns(ws) < m.numberOfSensesPerWord(wrd(ws)), "ws" + ws)

    syn(ws / m.ENCODE)(ws % m.ENCODE)(iemb) += newValue
  }

  //count the sense which is used
  def addSenseCount(sentence: Array[Int]): Unit = {
    for (ws <- sentence) {
      senseCountAdd(wrd(ws))(sns(ws)) += 1
    }
  }

  //skip-gram learning from the whole sentence
  def learnSentence(alpha: Float, sentence: Array[Int], sentenceNEG: Array[Array[Int]]): (Double, Int) = {
    var loss = 0.0
    var lossNum = 0
    for (pos <- 0 to sentence.size - 1) {
      val ws = sentence(pos)
      val lossWordSense = learnWordSense(ws, pos, alpha, sentence, sentenceNEG)
      loss += lossWordSense._1
      lossNum += lossWordSense._2
    }
    (loss, lossNum)
  }


  /**
    * compute derivative and update syn0, syn1 for word-sense wsInp at position pos
    * derivation of derivative see method getLossDeriv
    *
    * @param inpWs       word-sense indicator
    * @param pos         position of ws in the sentence
    * @param alpha       learningrate
    * @param sentence    sentence
    * @param sentenceNEG negative examples for word-senses of sentence
    * @return
    */
  def learnWordSense(inpWs: Int, pos: Int, alpha: Float, sentence: Array[Int], sentenceNEG: Array[Array[Int]]):
  (Double, Int) = {
    //val (syn0Old, syn1Old) = m.bdv2syn(m.syn2bdv(syn0, syn1)) // clone
    var loss = 0.0
    var lossNum = 0
    if (m.softMax) {
      for (p <- pos - m.window to pos + m.window) {
        if (p != pos && p >= 0 && p < sentence.size) {
          //val outWord = sentence(p) / m.ENCODE
          //val outSense = sentence(p) % m.ENCODE
          val outWs = if (m.syn1OneSense) wrdSns0(sentence(p)) else sentence(p)

          val x = m.dotProd(emb(syn0, inpWs), emb(syn1, outWs))
          val negs: Array[Int] = sentenceNEG(p)
          val y = new Array[Double](negs.length)
          for (j <- 0 until negs.length) {
            val negWs = negs(j)
            //val negWord = negWs / m.ENCODE
            //val negSense = negWs % m.ENCODE
            y(j) = m.dotProd(emb(syn0, inpWs), emb(syn1, negWs))
          }
          val (logSmxX, derivX, derivY) = logSoftmax(x, y, true)
          //println("outWs=" + outWs + " inpWs=" + inpWs + " softmx=" + logSmxX)
          loss += logSmxX

          for (i <- 0 until m.vectorSize) {
            embInc(syn1, sentence(p), i, -alpha * (derivX * emb(syn0, inpWs)(i)).toFloat)
            embInc(syn0, inpWs, i, -alpha * (derivX * emb(syn1, outWs)(i)).toFloat)
          }

          for (j <- 0 until negs.length) {
            for (i <- 0 until m.vectorSize) {
              embInc(syn1, negs(j), i, -alpha * (derivY(j) * emb(syn0, inpWs)(i)).toFloat)
              embInc(syn0, inpWs, i, -alpha * (derivY(j) * emb(syn1,negs(j))(i)).toFloat)
            }
          }
          lossNum += 1
        }
      }
      //    for (iw <- 0 until syn0.length)
      //      for (is <- 0 until syn0(iw).length)
      //        for (i <- 0 until syn0(iw)(is).length) {
      //          if (syn0(iw)(is)(i) != syn0Old(iw)(is)(i)) println("syn0" + iw + " " + is + " " + i)
      //          if (syn1(iw)(is)(i) != syn1Old(iw)(is)(i)) println("syn1" + iw + " " + is + " " + i)
      //        }
    } else {
      for (p <- pos - m.window to pos + m.window) {
        if (p != pos && p >= 0 && p < sentence.size) {
          val outWs = if (m.syn1OneSense) wrdSns0(sentence(p)) else sentence(p)
          var NEG: Array[Int] = sentenceNEG(p)
          val gradInp = new Array[Float](m.vectorSize)
          val sigmo = actFunSigmoidApprox(emb(syn0, inpWs), emb(syn1, outWs))
          val g = (1 - sigmo).toFloat
          loss += -math.log(sigmo)
          lossNum += 1
          //  y = a*x + y. x and y are n-dim. vectors :
          // int n, float sa, float[] sx, int incx, float[] sy, int incy
          // gradInp += g*syn1(word(outWs))(sense(outWs))
          blas.saxpy(m.vectorSize, g, emb(syn1, outWs), 1, gradInp, 1)
          // syn1(word(u))(sense(u)) += g*alphaU*syn0(word(u))(sense(u))
          blas.saxpy(m.vectorSize, g * alpha, emb(syn0, inpWs), 1, emb(syn1, outWs), 1)

          for (negWs <- NEG) { // if syn1OneSense then negatives always have sense 0
            val sigmoNeg = actFunSigmoidApprox(emb(syn0, inpWs), emb(syn1, negWs))
            val g = -sigmoNeg
            loss += -math.log(1 - sigmoNeg)
            lossNum += 1
            // gradInp += g * syn1(word(negWs))(sense(negWs))
            blas.saxpy(m.vectorSize, g, emb(syn1, negWs), 1, gradInp, 1)
            // syn1(word(negWs))(sense(negWs)) += g*alphaU*syn0(word(negWs))(sense(negWs))
            blas.saxpy(m.vectorSize, g * alpha, emb(syn0, inpWs), 1, emb(syn1, negWs), 1)
          }
          // syn0(word(ws))(sense(ws)) += alphaW*gradInp
          blas.saxpy(m.vectorSize, alpha, gradInp, 1, emb(syn0, inpWs), 1)
        }
      }
      //    for (iw <- 0 until syn0.length)
      //      for (is <- 0 until syn0(iw).length)
      //        for (i <- 0 until syn0(iw)(is).length) {
      //          if (syn0(iw)(is)(i) != syn0Old(iw)(is)(i)) println("syn0" + iw + " " + is + " " + i)
      //          if (syn1(iw)(is)(i) != syn1Old(iw)(is)(i)) println("syn1" + iw + " " + is + " " + i)
      //        }
    }
    (loss, lossNum)
  }


  /**
    * skip-gram model, use center word to predict surrounding words
    * compute derivative and update syn0, syn1 for word-sense wsInp at position pos
    * sig(x) = 1/(1+exp(-x)) = exp(x)/(1+exp(x)) German Wikipedia, checked by www.symbolab.com
    * 1-sig(-x) = 1 - 1/(1+exp(x)) = (1+exp(x)-1)/(1+exp(x)) = exp(x)/(1+exp(x)) =  sig(x)
    * i.e. 1-sig(x) = sig(-x)
    * d sig(x)/dx = d (1/(1+exp(-x))) / dx
    * = (1/(1+exp(-x)))**2 * d (1+exp(-x)) / dx
    * = (1/(1+exp(-x)))**2 * exp(-x) * (-1)
    * = (1/(1+exp(-x)))* (1/(1+exp(-x))) * (-exp(-x))
    * = (1/(1+exp(-x)))* (-exp(-x)/(1+exp(-x)))  // -exp(-x)/(1+exp(-x)) = -sig(-x) = 1-sig(x)
    * = sig(x) * (1-sig(x))
    *
    * LOSS-FUNCTION: formula (55) in Rong, should be minimized as then log sig(u'v) is large -> u'v is large
    * l(u,v) = -log sig(sum_j u_j*v_j) - sum_{w in NEG} log sig(- sum_j w_j*v_j)
    *
    * d-log sig(x)/dx = (-1)*1/sig(x) * sig(x)(1-sig(x)) = (-1)*(1-sig(x)) = sig(x)-1
    * d-log sig(sum_j u_j*v_j)/du_i = (sig(x)-1)*v_i
    * d-log sig(sum_j u_j*v_j)/dv_i = (sig(x)-1)*u_i
    *
    * d-log sig(-x)/dx =  (-1)/sig(-x) * (sig(-x)*(1-sig(-x))) *(-1) = 1-sig(-x) = sig(x)
    * d-log sig(-sum_j neg_j*inp_j)/dneg_i = sig(x)*inp_i
    * d-log sig(-sum_j neg_j*inp_j)/dinp_i = sig(x)*neg_i
    *
    * @param inpWs       word-sense indicator of word in sentence(pos)
    * @param pos         position of ws in the sentence
    * @param sentence    sentence
    * @param sentenceNEG negative examples for word-senses of sentence
    * @return
    */
  def getLossDerivSigmoid(inpWs: Int, pos: Int, /*alphaW: Float, alphaU: Float,*/ sentence: Array[Int], sentenceNEG:
  Array[Array[Int]], exact: Boolean): (Double, Array[Float]) = {
    val synInp = syn0
    val synOut = syn1
    val debug = false
    val drvInp = m.initSynZero()
    val drvOut = m.initSynZero()
    val (drInp, drOut) =
      if (debug) (m.initSynZero(), m.initSynZero()) // initialize derivatives to 0
      else (null, null)
    var loss = 0.0
    var lossNum = 0
    require(0 <= wrd(inpWs) && wrd(inpWs) < m.vocabSize, "NOT 0<=inpWord && inpWord < m.vocabSize")
    require(0 <= sns(inpWs) && sns(inpWs) < m.numberOfSensesPerWord(wrd(inpWs)))

    for (p <- pos - m.window to pos + m.window) {
      if (p != pos && p >= 0 && p < sentence.size) {
        val outWs = if (m.syn1OneSense) wrdSns0(sentence(p)) else sentence(p)
        val gradInp = new Array[Float](m.vectorSize)
        //val l = actFunSigmoidApprox(synInp(wsInp / m.ENCODE)(wsInp % m.ENCODE), synOut(u / m.ENCODE)(u % m.ENCODE),false)
        val sigmoid =
          if (exact) actFunSigmoid(emb(synInp, inpWs), emb(synOut, outWs))
          else actFunSigmoidApprox(emb(synInp, inpWs), emb(synOut, outWs))
        if (debug)
          println("SIGMOID: outWs=" + outWs + " inpWs=" + inpWs + " sigmoid=" + sigmoid)
        val g = (sigmoid - 1).toFloat
        loss += -math.log(sigmoid)
        lossNum += 1
        //val outWord = outWs / m.ENCODE;
        require(0 <= wrd(outWs) && wrd(outWs) < m.vocabSize)
        //val outSense = outWs % m.ENCODE;
        require(0 <= sns(outWs) && sns(outWs) < m.numberOfSensesPerWord(wrd(outWs)))
        //  sy = sa*sx + sy. sx and sy are n-dim. vectors :
        // int n, float sa, float[] sx, int incx, float[] sy, int incy
        // gradInp += g*synOut(word(u))(sense(u))
        blas.saxpy(m.vectorSize, g, emb(synOut, outWs), 1, gradInp, 1)
        if (debug)
          for (i <- 0 until m.vectorSize) {
            embInc(drOut, outWs, i, ((sigmoid - 1.0) * (emb(synInp, inpWs)(i))).toFloat)
            embInc(drInp, inpWs, i, ((sigmoid - 1.0) * emb(synOut, outWs)(i)).toFloat)
          }

        // synOut(word(u))(sense(u)) += g*alphaU*synInp(inpWord)(inpSense)  // derivative of synOut
        //blas.saxpy(m.vectorSize, g * alphaU, synInp(inpWs / m.ENCODE)(inpWs % m.ENCODE), 1, synOut(u / m.ENCODE)(u % m
        //  .ENCODE), 1)
        blas.saxpy(m.vectorSize, g, emb(synInp, inpWs), 1, emb(drvOut, outWs), 1)

        val NEG: Array[Int] = sentenceNEG(p)


        for (negWs <- NEG) { // if syn1OneSense then the negatives always have sense 0
          require(!m.syn1OneSense || sns(negWs)==0,"NOT !m.syn1OneSense || sns(negWs)==0")
          //val l = actFunSigmoidApprox(synInp(wsInp / m.ENCODE)(wsInp % m.ENCODE), synOut(negWs / m.ENCODE)(negWs % m.ENCODE),neg)
          val sigmoNeg: Float =
            if (exact) actFunSigmoid(emb(synInp, inpWs), emb(synOut, negWs))
            else actFunSigmoidApprox(emb(synInp, inpWs), emb(synOut, negWs))
          val g = sigmoNeg
          loss += -math.log(1 - sigmoNeg) // 1-sig(-x) =  sig(x), i.e. 1-sig(x) = sig(-x)
          lossNum += 1
          if (debug)
            println("negWs=" + negWs + " inpWs=" + inpWs + " sigmoNeg=" + sigmoNeg)
          //val negWord = negWs / m.ENCODE
          //val negSense = negWs % m.ENCODE
          // gradInp += g * synOut(word(negWs))(sense(negWs))
          blas.saxpy(m.vectorSize, g, emb(synOut, negWs), 1, gradInp, 1)
          // synOut(word(negWs))(sense(negWs)) += g*alphaU*synInp(word(negWs))(sense(negWs))
          //blas.saxpy(m.vectorSize, g * alphaU, synInp(wsInp / m.ENCODE)(wsInp % m.ENCODE), 1, synOut(negWs / m.ENCODE)
          // (negWs % m.ENCODE), 1)
          blas.saxpy(m.vectorSize, g, emb(synInp, inpWs), 1, emb(drvOut, negWs), 1)
          if (debug)
            for (i <- 0 until m.vectorSize) {
              embInc(drOut, negWs, i, (sigmoNeg * emb(synInp, inpWs)(i)).toFloat)
              embInc(drInp, inpWs, i, (sigmoNeg * emb(synOut, negWs)(i)).toFloat)
            }

        }

        // synInp(word(ws))(sense(ws)) += alphaW*gradInp
        // blas.saxpy(m.vectorSize, alphaW, gradInp, 1, synInp(wsInp / m.ENCODE)(wsInp % m.ENCODE), 1)
        blas.saxpy(m.vectorSize, 1.0f, gradInp, 1, emb(drvInp, inpWs), 1)
      }
    }

    val drv = m.stackSynIntoArray(drvInp) ++ m.stackSynIntoArray(drvOut)
    if (debug) {
      val dr = m.stackSynIntoArray(drInp) ++ m.stackSynIntoArray(drOut)
      for (iw <- 0 until drvInp.length) {
        for (is <- 0 until drvInp(iw).length) {
          for (i <- 0 until drvInp(iw)(is).length)
            println(iw + " " + is + " drvInp " + drvInp(iw)(is)(i) + " drInp " + drInp(iw)(is)(i) + " drvOut " +
              drvOut(iw)(is)(i)
              + " drOut " + drOut(iw)(is)(i))
        }
      }

      for (j <- 0 until drv.length) require(math.abs(drv(j) - dr(j)) < 1e-6, "NOT j=" + j)
    }
    (loss, drv)
  }


  /**
    * skip-gram model, use center word to predict surrounding words
    *
    * loss = - log(softmax(x,y)) where
    * where x = sum_j u_j*v_j
    * y_i = sum_j w_j*v_j for  y_i in NEG
    *
    * d -log(z(u))/du = -1/log(z(u)) * dz(u)/du
    *
    * @param inpWs       word-sense indicator of word in sentence(pos)
    * @param pos         position of ws in the sentence
    * @param sentence    sentence
    * @param sentenceNEG negative examples for word-senses of sentence
    * @return
    */
  def getLossDerivLogSoftmax(inpWs: Int, pos: Int, sentence: Array[Int], sentenceNEG: Array[Array[Int]],
                             exact: Boolean):
  (Double, Array[Float]) = {
    val synInp = syn0
    val synOut = syn1
    val debug = false
    val drvInp = m.initSynZero()
    val drvOut = m.initSynZero()
    var loss = 0.0
    var lossNum = 0
    //val inpWord = inpWs / m.ENCODE
    require(0 <= wrd(inpWs) && wrd(inpWs) < m.vocabSize, "NOT 0<=inpWord && inpWord < m.vocabSize")
    //val inpSense = inpWs % m.ENCODE;
    require(0 <= sns(inpWs) && sns(inpWs) < m.numberOfSensesPerWord(wrd(inpWs)))
    for (p <- pos - m.window to pos + m.window)
      if (p != pos && p >= 0 && p < sentence.size) {
        val outWs = sentence(p)
        //val outWord = outWs / m.ENCODE
        //val outSense = outWs % m.ENCODE

        val x = m.dotProd(emb(synInp, inpWs), emb(synOut, outWs))
        val negs: Array[Int] = sentenceNEG(p)
        val y = new Array[Double](negs.length)
        for (j <- 0 until negs.length) {
          val negWs = negs(j)
          y(j) = m.dotProd(emb(synInp, inpWs), emb(synOut, negWs))
        }
        val (logSmxX, derivX, derivY) = logSoftmax(x, y, true)
        println("outWs=" + outWs + " inpWs=" + inpWs + " softmx=" + logSmxX)
        loss += logSmxX

        require(0 <= wrd(outWs) && wrd(outWs) < m.vocabSize)
        require(0 <= sns(outWs) && sns(outWs) < m.numberOfSensesPerWord(wrd(outWs)))
        for (i <- 0 until m.vectorSize) {
          embInc(drvOut, outWs, i, (derivX * emb(synInp, inpWs)(i)).toFloat)
          embInc(drvInp, inpWs, i, (derivX * emb(synOut, outWs)(i)).toFloat)
        }

        for (j <- 0 until negs.length) {
          val negWs = negs(j)
          for (i <- 0 until m.vectorSize) {
            embInc(drvOut, negWs, i, (derivY(j) * emb(synInp, inpWs)(i)).toFloat)
            embInc(drvInp, inpWs, i, (derivY(j) * emb(synOut, negWs)(i)).toFloat)
          }
        }
        lossNum += 1

      }
    val drv = m.stackSynIntoArray(drvInp) ++ m.stackSynIntoArray(drvOut)
    (loss, drv)
  }

  /** sigmoid function applying on two vectors
    * compute exp(v0'v1)/(exp(v0'v1)+1). the sigmoid is approximated by a table
    *
    * @param v0
    * @param v1
    * @return
    */
  @inline final private def actFunSigmoidApprox(v0: Array[Float], v1: Array[Float]): Float = {
    val vectorSize = v0.length
    // f = v0^T * v1
    val f = blas.sdot(vectorSize, v0, 1, v1, 1)
    val res = if (f > m.MAX_EXP)
      m.expTable(m.expTable.length - 1)
    else if (f < -m.MAX_EXP)
      m.expTable(0)
    else {
      val ind = ((f + m.MAX_EXP) * (m.expTable.size / m.MAX_EXP / 2.0)).toInt
      m.expTable(ind)
    }
    res
  }

  //sigmoid function applying on two vectors
  /**
    * compute exp(v0'v1)/(exp(v0'v1)+1). the sigmoid is approximated by a table
    *
    * @param v0
    * @param v1
    * @return
    */
  @inline final private def actFunSigmoid(v0: Array[Float], v1: Array[Float]): Float = {
    val vectorSize = v0.length
    // f = v0^T * v1
    val f = blas.sdot(vectorSize, v0, 1, v1, 1)
    sigmoid(f)
  }

  /** compute approximate sigmoid
    *
    * @param x input value
    * @return exp(x)/(exp(x) +1) approximate
    */
  @inline final def sigmoidApprox(x: Float): Float = {
    if (x > m.MAX_EXP)
      m.expTable(m.expTable.length - 1)
    else if (x < -m.MAX_EXP)
      m.expTable(0)
    else {
      val ind = ((x + m.MAX_EXP) * (m.expTable.size / m.MAX_EXP / 2.0)).toInt
      m.expTable(ind)
    }
  }


  /** compute  sigmoid = exp(x)/(exp(x)+1.0)
    * x = small -> sigmoid ~ 0.0
    * x = large -> sigmoid ~ 1.0
    *
    * @param x input value
    * @return exp(x)/(exp(x) +1)
    */
  @inline final def sigmoid(x: Float): Float = {
    val f = 1.0 / (1.0 + math.exp(-x))
    f.toFloat
  }


  /**
    * softmax function
    * exp(x) / (exp(x) + sum_j exp(y_j)) = exp(x-mx) / (exp(x-mx) + sum_j exp(y_j-mx))
    * with mx selectes as max(x,y_1,...,y_n)
    *
    * @param x
    * @param y
    */
  @inline final def softmax(x: Double, y: Array[Double]): Double = {
    var j = 0
    // find max value to make sure later that exponent is computable
    var mx = x
    for (j <- 0 until y.length)
      mx = math.max(mx, y(j))
    var expX = math.exp(x - mx)
    var sm = expX
    for (j <- 0 until y.length)
      sm += math.exp(y(j) - mx)
    expX - math.log(sm)
  }


  /**
    * log softmax function
    * - log(exp(x) / (exp(x) + sum_j exp(y_j))) = -log(exp(x-mx) / (exp(x-mx) + sum_j exp(y_j-mx)))
    * with mx selected as max(x, y_1,..., y_n)
    *
    * @param x
    * @param y
    * @return (softmaxX, derivX, derivY)
    */
  @inline final def logSoftmax(x: Double, y: Array[Double], deriv: Boolean):
  (Double, Double, Array[Double]) = {
    var j = 0
    // find max value to make sure later that exponent is computable
    var mx = x
    for (j <- 0 until y.length)
      mx = math.max(y(j), mx)
    val expMx = math.exp(mx) //

    if (!deriv) {
      var expX = math.exp(x - mx)
      var expSm = expX
      for (j <- 0 until y.length) {
        val expY = math.exp(y(j) - mx)
        expSm += expY
      }
      val softmaxX = expX / expSm
      (math.log(softmaxX), Double.NaN, new Array[Double](0))
    } else {
      val expY = new Array[Double](y.length)
      var expX = math.exp(x - mx)
      var expSum = expX
      for (j <- 0 until y.length) {
        expY(j) = math.exp(y(j) - mx)
        expSum += expY(j)
      }
      val softmaxX = expX / expSum
      val derivX = -(1.0 - softmaxX) // - 1.0/softmaxX *(softmaxX) * (1.0-softmaxX)

      val softmaxY = new Array[Double](y.length)
      val derivY = new Array[Double](y.length)
      for (j <- 0 until y.length) {
        softmaxY(j) = expY(j) / expSum
        derivY(j) = softmaxY(j) //-(1.0/softmaxX )*(-1.0)*softmaxY(j)*softmaxX
      }

      (-math.log(softmaxX), derivX, derivY)
    }

  }

}