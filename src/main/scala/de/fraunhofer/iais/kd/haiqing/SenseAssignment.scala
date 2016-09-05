package de.fraunhofer.iais.kd.haiqing

import java.io._
import java.nio.file.{Files, Paths, Path}
import java.util.Calendar

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{AccumulatorParam, SparkContext}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.StringBuilder
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.compat.Platform._
import scala.io.Source
import breeze.linalg.{*, CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, axpy => Baxpy, max => Bmax, min => Bmin, sum => Bsum}
import breeze.numerics.{ceil, abs => Babs, exp => Bexp, log => Blog, sigmoid => Bsigmoid}
import breeze.stats.distributions.Multinomial
import de.fraunhofer.iais.kd.util.AliasMethod

import collection.JavaConverters._
import scala.util.Random


/**
  * Created by hwang on 09.02.16.
  */

/**
  * @param inputFile          name of input file
  * @param numRDDs            number of RDDs to contain the data
  * @param numEpoch           number of times to go through the data
  * @param minCount           words with lower counts are omitted
  * @param freqThreshStr      string containing thresholds for the counts of words separated by "_"
  *                           if (freqThreshStr(i)<=count(words)< freqThreshStr(i+1)) then word gets i+2 senses
  * @param seed
  * @param local
  * @param validationIsSubset // true if validation set is subset of trainingset
  */
class SenseAssignment(val inputFile: String,
                      val numRDDs: Int,
                      var numEpoch: Int,
                      val minCount: Int,
                      val freqThreshStr: String,
                      var seed: Long,
                      val local: Boolean,
                      var validationRatio: Float = 0.1f,
                      val maxValidationSize: Int = 10000,
                      val validationIsSubset: Boolean = false,
                      val readParamsFromFile: Boolean = false
                     ) extends Serializable {
  println("---- SenseAssignment parameters \n freqThreshStr=" + freqThreshStr + "\n seed=" + seed + "\n local=" + local+
    "\n validationRatio=" + validationRatio + "\n maxValidationSize=" + maxValidationSize + "\n " +
    "\n validationIsSubset=" + validationIsSubset+ "\n readParamsFromFile="+readParamsFromFile)
  var mc: ModelConst = null
  //private var vectorSize = 100
  //val minCount = freqThresh0(0)
  private var minCountMultiSense = 1000
  val count2numSenses = freqThreshStr.split("_").map(s => s.toInt).toArray

  private var testStep = 5
  val POWER = 0.75
  private val FREQ_LOOKUP_TABLE_SIZE = 100000
  // must be much larger than than vocab size TODO
  private val U = 100
  //private val ENCODE = 100
  // maximum number of senses per word
  //private val expTable = createExpTable()

  def getMAX_EXP = mc.MAX_EXP


  var nparam = 0 // number of free parameters


  var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]
  //private var numberOfSensesPerWord: Array[Int] = null
  private var totalWords = 0l
  //private var totalSentences = 0
  private var syn0: Array[Array[Array[Float]]] = null
  private var syn1: Array[Array[Array[Float]]] = null
  private var trainingSet: RDD[Array[Int]] = null
  private var validationSet: RDD[(Array[Int], Array[Array[Int]])] = null
  //private var smoothedFrequencyLookupTable: Array[Int] = null
  private var multinomFreqDistr: AliasMethod = null
  //private var sc: SparkContext = null

  def setModelConst(numNegative: Int, window: Int, vectorSize: Int, learningRate: Float, ENCODE: Int,
                    gamma: Float, oneSense: Boolean, softMax: Boolean, evaluationWordPath: String, modelPathOneSense: String,
                    modelPathMultiSense: String, modelSaveIter: Int, modelValidateIter: Int, maxEmbNorm: Float,
                    senseProbThresh: Float, printLv: Int, weightDecay: Float,syn1OneSense:Boolean): Unit = {
    require(mc == null, "NOT mc==null")
    mc = new ModelConst(window, vectorSize, count2numSenses.length + 1, numNegative, learningRate, ENCODE, gamma,
      oneSense, softMax, evaluationWordPath, modelPathOneSense, modelPathMultiSense, modelSaveIter, modelValidateIter, maxEmbNorm,
      senseProbThresh, printLv, weightDecay,syn1OneSense)
  }

  def setMinCountMultiSense(minCountMultiSense: Int): this.type = {
    this.minCountMultiSense = minCountMultiSense
    this
  }

  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  def setTestStep(testStep: Int): this.type = {
    this.testStep = testStep
    this
  }

  def setValidationRatio(validationRatio: Float): this.type = {
    this.validationRatio = validationRatio
    this
  }

  /**
    * generates values of x -> exp(x)/(exp(x)+1)
    * i=0 -> exp(-6)/(exp(-6)+1)  and i=EXP_TABLE_SIZE-1  -> exp(6)/(exp(6)+1)
    *
    * @return
    */


  //  /**
  //    *
  //    * generates an in array which contains a number of entries for every word according to frequency(word) to POWER
  //    *
  //    * @return
  //    */
  //  def createSmoothedFrequencyLookupTable(): Array[Int] = {
  //    require(mc.vocabSize * 5 < FREQ_LOOKUP_TABLE_SIZE, "NOT vocabSize*5<FREQ_LOOKUP_TABLE_SIZE")
  //    val table = new Array[Int](FREQ_LOOKUP_TABLE_SIZE)
  //    var trainWordsPow = 0.0
  //    val maxPow = Math.pow(vocab(0).cn, POWER)
  //    val minPow = Math.pow(vocab(mc.vocabSize - 1).cn, POWER)
  //    var ntab = 0
  //    for (a <- 0 until mc.vocabSize) {
  //      trainWordsPow += Math.pow(vocab(a).cn, POWER) // POWER = 0.75
  //      ntab += (Math.pow(vocab(a).cn, POWER) / minPow).floor.toInt
  //    }
  //    println("required FREQ_LOOKUP_TABLE_SIZE=" + ntab)
  //    require(ntab <= FREQ_LOOKUP_TABLE_SIZE, "NOT ntab<= FREQ_LOOKUP_TABLE_SIZE")
  //    var wordIndex = 0
  //    var d1 = Math.pow(vocab(wordIndex).cn, POWER) / trainWordsPow
  //    for (a <- 0 until FREQ_LOOKUP_TABLE_SIZE) {
  //      table(a) = wordIndex
  //      if (a * 1.0 / FREQ_LOOKUP_TABLE_SIZE > d1) {
  //        wordIndex += 1
  //        d1 += Math.pow(vocab(wordIndex).cn, POWER) / trainWordsPow
  //      }
  //      if (wordIndex >= mc.vocabSize) // low frequency words get same wordIndex
  //        wordIndex = mc.vocabSize - 1
  //    }
  //    require(table.toList.distinct.length == mc.vocabSize,
  //      table.toList.distinct.length + "=table.toList.distinct.length!=vocabSize=" + mc.vocabSize)
  //    table
  //  }

  def createMultinomFreqDistr(): AliasMethod = {
    val prob = new java.util.ArrayList[java.lang.Double](mc.vocabSize) // that as input for java
    var sm = 0.0
    for (iw <- 0 until mc.vocabSize) {
      val vl = math.pow(vocab(iw).cn * 1.0, POWER)
      prob.add(vl)
      sm += vl
    }
    for (iw <- 0 until mc.vocabSize) {
      prob.set(iw, prob.get(iw) / sm)
    }

    val multinomFrqDistr = new AliasMethod(prob)
    multinomFrqDistr
  }

  @deprecated("no longer needed", "01-05-2016")
  def createMultinomFreqDistrAlternative(rand: Random): UtMultinomial = {
    val prob = new Array[Double](mc.vocabSize)
    var sm = 0.0
    for (iw <- 0 until mc.vocabSize) {
      val vl = math.pow(vocab(iw).cn * 1.0, POWER)
      prob(iw) = vl
      sm += vl
    }
    for (iw <- 0 until mc.vocabSize) {
      prob(iw) = prob(iw) / sm
    }

    val multinomFrqDistr = new UtMultinomial(prob, rand)
    multinomFrqDistr
  }

  //  def TrainOneSense(input: RDD[String], outputPath: String): Unit = {
  //    val startTime = currentTime
  //
  //    val rand = new util.Random(seed)
  //
  //    learnVocab(input) // generates vocabulary
  //
  //    val (vocabSize, numberOfSensesPerWord) = createSenseTable(rand) // senseTable contains the number of senses for each
  //    mc.setDictInfo(vocabSize, numberOfSensesPerWord, createMultinomFreqDistr())
  //
  //    initSynLocalRandomly(rand) // init embeddings
  //
  //    // input still has line structure
  //    val splittedData = input.randomSplit(Array[Double](1 - validationRatio, validationRatio), rand.nextLong())
  //
  //    trainingSet = makeSentences(splittedData(0), rand) //split lines to tokens and preprocess tokens, convert to indices
  //    println("created training set with " + trainingSet.count().toInt)
  //
  //    validationSet = makeSentencesWithNEG(splittedData(1), rand).cache()
  //    println("created validation set with " + validationSet.count().toInt)
  //
  //    train()
  //
  //    writeToFile(outputPath)
  //
  //    println("total time:" + (currentTime - startTime) / 1000.0)
  //  }

  /**
    * training with multiple senses
    *
    * @param input rdd with lines of input
    *
    *
    */
  def trainWrapper(input: RDD[String]): Unit = {
    val startTime = currentTime
    require((new File(mc.modelPathOneSense)).exists(), "file does not exist: modelPathOneSense: "
      + mc.modelPathOneSense)
    if (mc.modelPathMultiSense.length > 0)
      require((new File(mc.modelPathMultiSense)).exists(), "file does not exist: modelPathMultiSense: " +
        mc.modelPathMultiSense)
    val numSentenceInput = input.count()

    val nSense = if (mc.oneSense) "oneSense" else "multiSense"
    println(nSense + " starting to train " + nSense + " at " + Calendar.getInstance().getTime())
    println("numSentences of input file " + numSentenceInput)
    val rand = new util.Random(seed)

    val valiRatio = math.min(validationRatio, maxValidationSize * 1.0f / numSentenceInput)
    val splittedData = input.randomSplit(Array[Double](1 - valiRatio, valiRatio), rand.nextLong())

    learnVocab(splittedData(0))
    //learnVocab(input)

    val (vocabSize, numberOfSensesPerWord) = createNumberOfSensePerWord(rand)
    mc.setDictInfo(vocabSize, numberOfSensesPerWord, this.createMultinomFreqDistr())

    trainingSet = makeSentences(splittedData(0), rand)
    validationSet = makeSentencesWithNEG(splittedData(1), rand).cache()

    if (mc.oneSense) {
      if (readParamsFromFile)
        initSynFromFile(mc.modelPathOneSense, rand, mc.senseInitStandardDev) // init embeddings with stored embedding
      else
        initSynLocalRandomly(rand) // init embeddings randomly
    } else {
      if (readParamsFromFile)
        initSynFromFile(mc.modelPathMultiSense, rand, mc.senseInitStandardDev) // init embeddings with stored embedding
      else
        initSynFromFile(mc.modelPathOneSense, rand, mc.senseInitStandardDev) // init embeddings with stored embedding
    }

//    // wih 1 sense
//    val valiRatio = math.min(validationRatio, maxValidationSize * 1.0f / numSentenceInput)
//    println(nSense + " validationIsSubset=" + validationIsSubset + " numSentenceInput=" + numSentenceInput
//      + " validationRatio=" + validationRatio + " maxValidationSize=" + maxValidationSize)
//    if (validationIsSubset) {
//      //validationset as subset of trainingset
//      trainingSet = makeSentences(input, rand)
//      validationSet = makeSentencesWithNEG(input.sample(false, valiRatio, 1357), rand).cache()
//    } else {
//      val splittedData = input.randomSplit(Array[Double](1 - valiRatio, valiRatio), rand.nextLong())
//      trainingSet = makeSentences(splittedData(0), rand)
//      validationSet = makeSentencesWithNEG(splittedData(1), rand).cache()
//    }


    println(nSense + " created   training set with " + trainingSet.count().toInt + " lines")
    println(nSense + " created validation set with " + validationSet.count().toInt + " lines\n")

    train()

    println(nSense + " total time:" + (currentTime - startTime) / 1000.0)
    println(nSense + " finished training at " + Calendar.getInstance().getTime())
  }

  def learnVocab(input: RDD[String]): Unit = {
    require(vocab == null, "NOT vocab==null")
    //remove the beginning and end non-letter, and then transform to lowercase letter
    val words = input
      .map(line => line.split(" ").array) // create tokens
      .map(sentence => cleanTokens(sentence)) // clean tokens of sentences
      .flatMap(x => x) // create one long sequence of tokens

    //      flatMap(x => x).filter(x => x.size > 0).map { x =>
    //      var begin = 0;
    //      var end = x.size - 1;
    //      while (begin <= end && !x(begin).isLetter)
    //        begin += 1
    //      while (begin <= end && !x(end).isLetter)
    //        end -= 1
    //      x.substring(begin, end + 1)
    //    }.map(x => x.toLowerCase).filter(x => x.size > 0)

    require(words != null, "words RDD is null. You may need to check if loading data correctly.")
    //build vocabulary with count of each word, and remove the infrequent words
    val vocab1 = words.map(w => (w, 1))
      .reduceByKey(_ + _)
    val vocabRaw = vocab1
      .map(x => VocabWord(
        x._1,
        x._2))
    val countFreq = vocabRaw.map(x => (x.cn, 1)).reduceByKey(_ + _).collect().sortWith((a, b) => a._1 < b._1)
    val accFreq = new Array[Int](countFreq.length)
    accFreq(countFreq.length - 1) = countFreq(countFreq.length - 1)._2
    println("---- frequency of word counts -------")
    for (i <- countFreq.length - 2 to 0 by -1)
      accFreq(i) = countFreq(i)._2 + accFreq(i + 1)

    //val diff = accFreq(0) / 400;
    val diff = 100
    var acc = 0
    for (i <- 0 until countFreq.length) {
      acc += countFreq(i)._2
      if (i == 0 || acc >= diff) {
        println("count=" + countFreq(i)._1 + " frequency=" + countFreq(i)._2 + " accumutated freq=" + accFreq(i))
        acc = 0
      }
    }

    val sc = input.context
    val evaluationWords = sc.textFile(mc.evaluationWordPath).collect().toSet

    vocab = vocabRaw
      .filter(x => (x.cn >= minCount || evaluationWords.contains(x.word)))
      .collect() // this executes the
      .sortWith((a, b) => a.cn > b.cn)
    mc.vocabSize = vocab.length
    require(mc.vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences. ")

    for (a <- 0 until mc.vocabSize) {
      vocabHash += vocab(a).word -> a
      totalWords += vocab(a).cn
    }

    println("raw vocabSize=" + vocab1.count + " selected vocabSize=" + mc.vocabSize + " ( count >= minCount=" +
      minCount + ")")
    println("number of running tokens: totalWords = " + totalWords)
  }

  /**
    * if count(word)>= count2numSenses(i) the assign i+2 senses. i.e. count(word)>= count2numSenses(0) => 2 senses
    * else 1 sense
    *
    * @param rand
    * @return
    */
  def createNumberOfSensePerWord(rand: util.Random): (Int, Array[Int]) = {
    require(vocab.length > 0, "NOT vocab.length>0")
    println("minCount=" + minCount + " count2numSenses=[" + count2numSenses.mkString(" ") + "]")
    require(count2numSenses.length == mc.maxNumSenses - 1,
      count2numSenses.length + "count2numSenses.length != mc.maxNumSenses-1" +
        (mc.maxNumSenses - 1))
    val numberOfSensesPerWord = new Array[Int](mc.vocabSize)
    val senseCounts = new Array[Int](count2numSenses.length + 1)

    println("---- print some words in vocabulary: ")
    for (a <- 0 until mc.vocabSize) {
      numberOfSensesPerWord(a) = 1
      if (!mc.oneSense) {
        for (i <- 0 until count2numSenses.length) {
          if (vocab(a).cn >= count2numSenses(i))
            numberOfSensesPerWord(a) = i + 2
        }
        require(numberOfSensesPerWord(a) <= mc.maxNumSenses)
      }
      if (rand.nextInt(mc.vocabSize) < 50)
        println(
          vocab(a).word + " wordIndex=" + a + " wordCount=" + vocab(a).cn + "  numSense=" + numberOfSensesPerWord(a))
      senseCounts(numberOfSensesPerWord(a) - 1) += 1
    }
    println("number of senseCounts=[" + senseCounts.mkString(" ") + "]")
    println("vocabSize = " + mc.vocabSize + "   running number of tokens = " + totalWords)
    println("-------------------------------------")
    (mc.vocabSize, numberOfSensesPerWord)
  }

  //  private def getNEG(w: Int, smoothedFrequencyLookupTable: Array[Int], rand: util.Random): Array[Int] = {
  //    val negSamples = new Array[Int](mc.numNegative)
  //    val tableSize = smoothedFrequencyLookupTable.size
  //    for (i <- 0 until mc.numNegative ) {
  //      negSamples(i) = w
  //      while (negSamples(i) / mc.ENCODE == w / mc.ENCODE) {
  //        negSamples(i) = smoothedFrequencyLookupTable(Math.abs(rand.nextLong() % tableSize).toInt)
  //        if (negSamples(i) <= 0)
  //          negSamples(i) = (Math.abs(rand.nextLong()) % (vocabSize - 1) + 1).toInt
  //      }
  //      //add sense information (assign sense randomly)
  //      negSamples(i) = negSamples(i) * ENCODE + rand.nextInt(numberOfSensesPerWord(negSamples(i)))
  //    }
  //    negSamples
  //  }

  /**
    * split lines to tokens and preprocess tokens
    * create the indices of word-sense combination. Assumes maximum number of senses per word
    *
    * @param input
    * @return
    */
  def makeSentences(input: RDD[String], rand: util.Random): RDD[Array[Int]] = {
    require(mc.vocabSize > 0, "The vocabulary size should be > 0. You may need to check if learning vocabulary " +
      "correctly.")

    val sc = input.context
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcNumberOfSensesPerWord = sc.broadcast(mc.numberOfSensesPerWord)
    val bcENCODE = sc.broadcast(mc.ENCODE)

    val sentenceRDD = input.map(line => line.split(" ").array) // split line into array of tokens
      .map { sentence => {
        val cleanedTokens = cleanTokens(sentence)
        val newSentence = cleanedTokens
          .filter(x => x.size > 0 && bcVocabHash.value.contains(x)) // remove tokens not in vocabulary
          /* assign a word-sense index with random sense */
          .map { x =>
          val wordId: Int = bcVocabHash.value.get(x).getOrElse(-1)
          //val wordId: Int = bcVocabHash.value.get(x).get
          val senseId = rand.nextInt(bcNumberOfSensesPerWord.value(wordId))
          wordId * bcENCODE.value + senseId
        }
        newSentence
       }
    }.filter(sentence=>sentence.size>0).cache()
    sentenceRDD
  }

  /**
    * preprocesses all tokens. has to be used in createVocabulary and makeSentences
    *
    * @param sentence string of raw tokens
    * @return
    */
  def cleanTokens(sentence: Array[String]): Array[String] = {
    val newSentence
    = sentence.filter(x => x.size > 0) // remove tokens of length 0
      .map { x =>
      var begin = 0
      var end = x.size - 1
      while (begin <= end && !x(begin).isLetter) //remove non-letters at begin or end of token
        begin += 1
      while (begin <= end && !x(end).isLetter)
        end -= 1
      x.substring(begin, end + 1)
    }
      .map(x => x.toLowerCase) // change tokens to lower case
      .filter(x => x.size > 0)
    newSentence
  }


  /**
    * create the indices of word-sense combination. Assumes maximum number of senses per word
    *
    * @param input
    * @return
    */
  private def makeSentencesWithNEG(input: RDD[String], rand: util.Random): RDD[(Array[Int], Array[Array[Int]])] = {
    require(mc.vocabSize > 0, "The vocabulary size should be > 0. You may need to check if learning vocabulary " +
      "correctly.")

    val sc = input.context
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcNumberOfSensesPerWord = sc.broadcast(mc.numberOfSensesPerWord)
    val bcENCODE = sc.broadcast(mc.ENCODE)

    val sentenceRDD = input.map(line => line.split(" ").array)
      .map { sentence => {
        val cleanedTokens = cleanTokens(sentence)
        val newSentence = cleanedTokens
          .filter(x => x.size > 0 && bcVocabHash.value.contains(x)) // remove tokens not in vocabulary
          /* assign a word-sense index with random sense */
          .map { x =>
          val wordId: Int = bcVocabHash.value.get(x).getOrElse(-1)
          //val wordId: Int = bcVocabHash.value.get(x).get
          val senseId = rand.nextInt(bcNumberOfSensesPerWord.value(wordId))
          wordId * bcENCODE.value + senseId
        }
        val sentenceNEG = newSentence.map(w => mc.getNEG(w, rand))
        (newSentence, sentenceNEG)
      }
    }.filter(sentenceToken=>sentenceToken._1.size>0).cache()
    sentenceRDD
  }

  //initialize from normal skip-gram model
  def initSynFromFile(synPath: String, rand: util.Random, standardDev: Float = 0.01f): Unit = {

    syn0 = SenseAssignment.readSyn(vocabHash, synPath + "/syn0.txt", rand, standardDev, mc.numberOfSensesPerWord)
    syn1 = SenseAssignment.readSyn(vocabHash, synPath + "/syn1.txt", rand, standardDev, mc.numberOfSensesPerWord)
    require(syn0(0)(0).length == mc.vectorSize,syn0(0)(0).length+"=syn0(0)(0).length == mc.vectorSize="+ mc.vectorSize)
    require(syn1(0)(0).length == mc.vectorSize,syn1(0)(0).length+"=syn1(0)(0).length == mc.vectorSize="+ mc.vectorSize)

    //    val wordIndexOld = Source.fromFile(synPath + "/wordIndex.txt").getLines().toArray
    //    val numberOfSensesPerWordOld = new Array[Int](wordIndexOld.length)
    //    require(wordIndexOld.length == mc.vocabSize, "NOT mc.vocabSize")
    //    for (i <- 0 until wordIndexOld.length) {
    //      val wis = wordIndexOld(i).split("_")
    //      vocabHash.get(wis(0)) match {
    //        case Some(x) =>
    //          require(x == wis(1).toInt, "wrong key " + wis(1) + " of word " + x)
    //          numberOfSensesPerWordOld(wis(1).toInt) = wis(2).toInt
    //        case None => throw new RuntimeException(wis(0) + " word not in vocabulary")
    //      }
    //    }
    //    var nparamOld = 0
    //    for (iw <- 0 until mc.vocabSize)
    //      nparamOld += numberOfSensesPerWordOld(iw) * mc.vectorSize

    //    val syn0Old = Source.fromFile(synPath + "/syn0.txt").getLines().map(line => line.split(" ").toSeq).flatten
    //      .map(s => s.toFloat).toArray
    //    val syn1Old = Source.fromFile(synPath + "/syn1.txt").getLines().map(line => line.split(" ").toSeq).flatten
    //      .map(s => s.toFloat).toArray
    println("read syn0,syn1 model parameters from directory " + synPath)
  }

  //  def readSyn(synFileName: String, rand: Random, standardDev: Float = 0.01f): Array[Array[Array[Float]]] = {
  //    val lines0 = Source.fromFile(synFileName).getLines.map(line => line.split(" ")).toArray
  //    val wordInitialized = new Array[Boolean](mc.vocabSize)
  //    val syn = mc.initSynZero()
  //    for (lin <- lines0) {
  //      val word = lin(0)
  //      val iword = vocabHash.getOrElse(word, -1)
  //      require(iword >= 0, "word " + word + " not found")
  //      val isense = lin(1).toInt
  //      require(isense < mc.numberOfSensesPerWord(iword), "NOT isense< mc.numberOfSensesPerWord(iword)")
  //      require(lin.length == mc.vectorSize + 2, "NOT lin.length== mc.vectorSize+2 ")
  //      if (!wordInitialized(iword)) {
  //        for (isen <- 0 until mc.numberOfSensesPerWord(iword))
  //          for (j <- 0 until mc.vectorSize)
  //            syn(iword)(isen)(j) = lin(j + 2).toFloat + (rand.nextFloat() - 0.5f) * standardDev / mc.vectorSize
  //      } else {
  //        for (j <- 0 until mc.vectorSize)
  //          syn(iword)(isense)(j) = lin(j + 2).toFloat
  //      }
  //      wordInitialized(iword) = true
  //    }
  //    syn
  //  }

  //initialize randomly
  private def initSynLocalRandomly(rand: util.Random): Unit = {
    val (sy0, sy1) = initSynRand(rand)
    syn0 = sy0
    syn1 = sy1
  }


  //initialize randomly
  def initSynRand(rand: util.Random): (Array[Array[Array[Float]]], Array[Array[Array[Float]]]) = {
    val sy0 = mc.initSynRand(rand)
    val sy1 = mc.initSynRand(rand, sd = 0.5f)
    println("number of parameters =" + 2 * mc.getTotalNumberOfParams())
    (sy0, sy1)
  }


  def bdv2synLocal(parm: BDV[Double]) = {
    val (sy0, sy1) = mc.bdv2syn(parm: BDV[Double])
    syn0 = sy0
    syn1 = sy1
  }


  /**
    * (fac* a + (1-fac)*b)fac + (1-fac)*b = a*fac*fac + (1-fac)*fac*b +  (1-fac)*b = a*fac*fac + (1-fac*fac)*b
    *
    * @param senseCnts  counts of sense with sum cntSum
    * @param factor
    * @param senseProbs probability of senses to be updated (=0.99)
    * @return
    */
  def updateSenseProbs(senseCnts: Array[Array[Int]], factor: Double, senseProbs: Array[Array[Float]]) {
    require(0.5 < factor && factor <= 1.0, "NOT 0 <= factor && factor <= 1.0");
    val logFactor = math.log(factor)
    for (itok <- 0 until senseCnts.length) {
      val senseCnt = senseCnts(itok)
      val sensePrb = senseProbs(itok)
      val cntSum = senseCnt.foldLeft(0)((a, b) => a + b).toDouble
      if (cntSum > 0) {
        // token has been observed
        val sensePrbNew = senseCnt.map(x => x / cntSum)
        val factorN = math.exp(cntSum * logFactor)
        for (isense <- 0 until sensePrb.length)
          sensePrb(isense) = (sensePrb(isense) * factorN + sensePrbNew(isense) * (1 - factorN)).toFloat
      }

    }
  }


  /**
    * reset embedding of a sense with probability lower senseProbThresh
    *
    * reset to weighted randomly selected sense with probability larger than senseProbThresh
    *
    * @param senseProbs      probability of senses per word
    * @param syn0            , syn1 parameters
    * @param senseProbThresh if probability is lower then a sense is re-initialized
    * @return
    */
  def resetLowProbSenses(senseProbs: Array[Array[Float]], syn0: Array[Array[Array[Float]]],
                         syn1: Array[Array[Array[Float]]], senseProbThresh: Float, rand: Random,
                         senseInitStandardDev: Float): (Int, Int) = {
    val factor: Float = senseInitStandardDev / syn0(0)(0).length
    var nreset = 0
    var nMultiSense = 0
    for (itok <- 0 until senseProbs.length) {
      val sensePrb = senseProbs(itok)
      val nsense = sensePrb.length
      if (nsense > 1) {
        val prbSum = sensePrb.foldLeft(0f)((a, b) => a + b)
        for (isense <- 0 until nsense) // renormalize sense probabilities
          sensePrb(isense) = sensePrb(isense) / prbSum
        nMultiSense += sensePrb.length

        for (isense <- 0 until nsense) {
          if (sensePrb(isense) < senseProbThresh) {
            val senseHi = new Array[Int](nsense)
            val senseHiProb = new ArrayBuffer[Double](nsense)
            var nHi = 0
            for (isHi <- 0 until nsense) {
              if (sensePrb(isHi) >= senseProbThresh) {
                senseHi(nHi) = isHi
                nHi += 1
                senseHiProb += sensePrb(isHi)
              }
            }
            val multinom = new UtMultinomial(senseHiProb.toArray, rand)
            val jHiUse = multinom.sample()
            val senseUse = senseHi(jHiUse)

            for (i <- 0 until syn0(itok)(isense).length) {
              // reset to embedding with hi probability
              syn0(itok)(isense)(i) = syn0(itok)(senseUse)(i) + factor * (rand.nextFloat() - 0.5f)
              if(!mc.syn1OneSense)
                syn1(itok)(isense)(i) = syn1(itok)(senseUse)(i) + factor * (rand.nextFloat() - 0.5f)
            }
            nreset += 1
          }
        }
      }
    }
    (nMultiSense, nreset)
  }

  private def train(): Unit = {
    val saveRDD = false

    require(syn0 != null, "syn0 should not be null. You may need to check if initializing parameters correctly.")
    require(syn1 != null, "syn1 should not be null. You may need to check if initializing parameters correctly.")
    require(syn0.length == syn1.length, "NOT syn0.length==syn1.length")
    val numSentenceTrainSet = trainingSet.count()


    val trainSet = trainingSet.randomSplit(new Array[Double](numRDDs).map(x => x + 1.0))
    val rand = new Random(seed + 1773)

    var totalWordCount = 0 // running words
    val totalTrainWords = totalWords * numEpoch * (1 - validationRatio) // totalWords = number of running words in corpus
    println("----------------------")
    println("totalWords="+totalWords)
    println("numEpoch="+numEpoch)
    println("validationRatio="+validationRatio)
    println("totalTrainWords="+totalTrainWords)
    println("----------------------")
    val numIterations = numEpoch * numRDDs

    val senseProbs = new Array[Array[Float]](mc.vocabSize) // for each word the probability of senses
    for (w <- 0 until mc.vocabSize) {
      senseProbs(w) = new Array[Float](mc.numberOfSensesPerWord(w)) // length of array = number of senses
      for (s <- 0 until mc.numberOfSensesPerWord(w))
        senseProbs(w)(s) = 1.0f / mc.numberOfSensesPerWord(w) // init as uniform probability
    }

    val str = if (mc.oneSense) mc.modelPathOneSense else mc.modelPathMultiSense
    //val outputPath = str+"/"+Calendar.getInstance().getTime.toString.map(x=>if(x!=' '&&x!=':')x else '-').substring(4,19)+"-Para:"+numRDDs+","+numEpoch+","+minCount+","+mc.numNegative+","+mc.window+","+mc.vectorSize+","+freqThreshStr+","+mc.learningRate+","+mc.gamma+","+local+","+trainingSet.getNumPartitions
    val outputPath = str
    println("!!!!!!!!"+outputPath)
    val folderPath: Path = Paths.get(outputPath)
    if (!Files.exists(folderPath))
      Files.createDirectory(folderPath)

    //val file = new PrintWriter(new File("./tmp.txt"))
    // ModelConst contains all constant structures of the model
    val historyFileName: String = outputPath + "/history.txt"

    val historyWriter = new PrintWriter(new File(historyFileName))
    val histHeader = Array("it", "valiLossPerPredict", "valiNumAdjustPerSentence")
    val histPos = new mutable.HashMap[String, Int]()
    for (i <- 0 until histHeader.length)
      histPos.update(histHeader(i), i)
    historyWriter.write(histHeader.mkString(" ") + "\n")
    historyWriter.flush()

    val storageLevel = StorageLevel.DISK_ONLY // write rdds to disk

    var alpha = mc.learningRate
    var lastValiLoss = -1.0;

    /*----------------------------------------------------------------------- */
    /*            iteration over RDDs                                         */
    /*----------------------------------------------------------------------- */
    var it = 0
    var flag = false
    while (it < numIterations && !flag) {

      val indexRDD = it % numRDDs
      val stIter = "epoch= " + (it / numRDDs) + "/" + numEpoch + " iRDD=" + indexRDD + "/" + this.numRDDs +
        " "
      val iterStart = System.currentTimeMillis()

      val hist = new Array[Double](histPos.size)
      hist(histPos("it")) = it


      //println("validationSet.count()=" + validationSet.count())

      if (!mc.oneSense) {
        /*----------------------------------------------------------------------- */
        /*            adjust senses of training set RDD                           */
        /*----------------------------------------------------------------------- */
        val scTrn1 = trainSet(indexRDD).context
        val mcBc = scTrn1.broadcast(mc)
        val syn0Bc = scTrn1.broadcast(syn0)
        val syn1Bc = scTrn1.broadcast(syn1)
        val senseCountAcc = scTrn1.accumulator(mc.initSenseCounts(mc.numberOfSensesPerWord))(CountArrAccumulatorParam)
        var numAdjustTrainAcc = scTrn1.accumulator(0)
        val modelConst1Bc = scTrn1.broadcast(mc)
        var numPartitions = trainSet(indexRDD).partitions.size
        val seedTrainAdjustBc = scTrn1.broadcast(seed + it * numPartitions + 9886) // always a different seed

        trainSet(indexRDD) = trainSet(indexRDD).mapPartitionsWithIndex { (idx, iter) =>
          val F = new ModelUpdater(modelConst1Bc.value, seedTrainAdjustBc.value + idx, syn0Bc.value, syn1Bc
            .value)
          val senseCount = F.m.initSenseCounts(F.m.numberOfSensesPerWord)
          // init senseCount
          val newIter = mutable.MutableList[Array[Int]]()
          for (sentence <- iter) {
            var nadj = 0
            val sentenceNEG = F.m.generateSentenceNEG(sentence, F.rand) // use same sentenceNEG for all sense assignment
            while (nadj < F.m.maxAdjusting) {
              val adjust = F.adjustSentence(sentence, sentenceNEG)
              if (adjust) nadj += 1 else nadj = F.m.maxAdjusting
              if (adjust) numAdjustTrainAcc += 1
            }
            for (pos <- 0 until sentence.length)
              senseCount(F.m.wrd(sentence(pos)))(F.m.sns(sentence(pos))) += 1
            newIter += sentence
          }
          senseCountAcc += senseCount
          newIter.toIterator
        }


        val bcVocab = scTrn1.broadcast(vocab)

        val sentenceNumTraining = trainSet(indexRDD).count()
        trainSet(indexRDD).persist(storageLevel)
        if (mc.printLv > 1)
          println(stIter + " trainRDD: numAdjustPerSentence: " + Ut.pp(numAdjustTrainAcc.value * 1.0
            / sentenceNumTraining, "%10.6f") + " trainRDD num sentence=" + sentenceNumTraining)

        //update senseProbs from senseCounts
        updateSenseProbs(senseCountAcc.value, 0.99, senseProbs)
        val (nMultiSense, nreset) = resetLowProbSenses(senseProbs, syn0, syn1, mc.senseProbThresh, rand,
          mc.senseInitStandardDev)
        if (mc.printLv > 1) {
          println(stIter + " trainRDD: numAdjustPerSentence: " + Ut.pp(numAdjustTrainAcc.value * 1.0
            / sentenceNumTraining, "%10.6f") + " trainRDD num sentence=" + sentenceNumTraining +
            " senseResetFraction=" + Ut.pp(nreset * 1.0 / nMultiSense, "%10.6f"))
        }


        //vocab.
        println("saveRDD="+saveRDD)
        if (saveRDD) {
          val StringRDD = trainSet(indexRDD).mapPartitionsWithIndex { (idx, iter) =>
            println(idx+"  savingRDD...")
            val F = new ModelUpdater(modelConst1Bc.value, seedTrainAdjustBc.value + idx, syn0Bc.value, syn1Bc
              .value)
            val newIter = mutable.MutableList[String]()
            for (sentence <- iter) {
              newIter += sentence.map(x=>bcVocab.value(F.m.wrd(x)).word+"_"+F.m.sns(x)).mkString(" ")
            }
            newIter.toIterator
          }.cache()
          println(StringRDD.count())
          StringRDD.saveAsTextFile(outputPath+"/trainRDD_"+indexRDD)
        }

      }

      println( "it="+it+"totalTrainWords="+totalTrainWords)

      val doTrain = true
      if (doTrain && !saveRDD) {

        /*----------------------------------------------------------------------- */
        /*            train parameters syn0 syn1 using 1 training set RDD         */
        /*----------------------------------------------------------------------- */

        //wordCount and numPartitions are for updating the alpha(learning rate)
        val scTrn2 = trainSet(indexRDD).context
        val syn0Bc = scTrn2.broadcast(syn0)
        val syn1Bc = scTrn2.broadcast(syn1)
        val mcBc = scTrn2.broadcast(mc)
        val wordCountTrainAcc = scTrn2.accumulator(0)
        val lossTrainAcc = scTrn2.accumulator(0.0)
        val lossNumTrainAcc = scTrn2.accumulator(0l)
        var numPartitions = trainSet(indexRDD).partitions.size
        val seedTrainBc = scTrn2.broadcast(seed + it * numPartitions + 178829)
        val alphaBc = scTrn2.broadcast(alpha)

        //println("iteration = " + it + "   indexRDD = " + indexRDD + " learn syn0 and syn1 ...")
        //------------------- learn syn0 and syn1 -------------------------
        val synRDD = trainSet(indexRDD).mapPartitionsWithIndex { (idx, iterator) =>
          // a different seed for each iteration and partition
          val F = new ModelUpdater(mcBc.value, seedTrainBc.value + idx, syn0Bc.value, syn1Bc.value)

          //update alpha
          var startTime = currentTime
          var lastWordCount = 0
          var wordCount = 0
          val alpha = alphaBc.value
//          var alpha = F.m.learningRate * (1 - totalWordCount * 1.0f / totalTrainWords)
//          if (alpha < F.m.learningRate * 0.0001f) alpha = F.m.learningRate * 0.0001f

          var loss = 0.0
          var lossNum = 0
          for (sentence <- iterator) {
//            if (wordCount - lastWordCount > F.m.stepSize) {
//              var alpha = F.m.learningRate * (1 - (totalWordCount * 1.0 + wordCount * numPartitions) / totalTrainWords)
//              if (alpha < F.m.learningRate * 0.0001f) alpha = F.m.learningRate * 0.0001f
//            }
            val sentenceNEG = F.m.generateSentenceNEG(sentence, F.rand)
            val (sentLoss, sentLossNum) = F.learnSentence(alpha.toFloat, sentence, sentenceNEG)
            loss += sentLoss
            lossNum += sentLossNum
            //about syn0Modify and syn1Modify may be a problem
            wordCount += sentence.size
          }

          if (mcBc.value.printLv > 0) {
            val wordPerSec = (wordCount - lastWordCount) * 1000 / (currentTime - startTime)
            var st = stIter + " partition=" + idx
            st += " lossPerPredict\t=" + Ut.pp(loss / lossNum, "%10.6f") + " numPredictions=" + lossNum
            st += " wordCount=" + (totalWordCount + wordCount * numPartitions)
            st += "/" + totalTrainWords + " wordsPerSec=" + wordPerSec
            st += " alpha=" + Ut.pp(alpha, "%10.6f")
            println(st)
            println("----------------------------")
            println("totalWordCount="+totalWordCount)
            println("wordCount="+wordCount)
            println("numPartitions="+numPartitions)
            println("----------------------------")
          }

          val synIter = new ArrayBuffer[(Int, Array[Float])]()
          val s0: (Int, Array[Float]) = (0, F.m.stackSynIntoArray(F.syn0))
          synIter += s0
          val s1: (Int, Array[Float]) = (1, F.m.stackSynIntoArray(F.syn1))
          synIter += s1
          //println("idx="+idx+" synIter.length="+synIter.length)
          synIter.toIterator
        }.cache()

        alpha = alpha*mc.gamma
        if (alpha < mc.learningRate*0.0001f)
          alpha = mc.learningRate*0.0001f

        val time1 = currentTime
        val nn = synRDD.count()
        println("count time="+(currentTime-time1)/1000.0)


        if (mc.printLv > 1) {
          var st = stIter + "trainSet: lossPerPrediction=" + Ut.pp(lossTrainAcc.value / lossNumTrainAcc.value, "%10.6f")
          st += " numPrediction: " + lossNumTrainAcc.value
          println(st)
          if (mc.printLv > 2)
            println("numPartitions=" + numPartitions + " synRDD.count()=" + nn)
        }
        require(nn == 2 * numPartitions, "NOT synRDD.count()==2*numPartitions")

        /*----------------------------------------------------------------------- */
        /*            aggregate the results from different partitions             */
        /*----------------------------------------------------------------------- */
        val time2 = currentTime
        val (syn0Avg, syn1Avg): (Array[Float], Array[Float]) =
          synRDD.treeAggregate(new Array[Float](0), new Array[Float](0))(
            // aggregate within RDD. c: aggregator, v: RDD-element
            seqOp = (c: (Array[Float], Array[Float]), v: (Int, Array[Float])) => {
              //println("treeagg seqOp: id=" + v._1 + " length=" + v._2.length)
              v._1 match {
                case 0 =>
                  require(c._1.length == 0, "NOT c._1.length==0")
                  (v._2, c._2)
                case 1 =>
                  require(c._2.length == 0, "NOT c._2.length==0")
                  (c._1, v._2)
              }
            },
            // aggregate between RDDs
            combOp = (c1: (Array[Float], Array[Float]), c2: (Array[Float], Array[Float])) => {
              //println("treeagg combOp: numParam=" + c1._1.length)
              require(c1._1.length == c2._1.length, "NOT c1._1.length==c2._1.length")
              require(c1._2.length == c2._2.length, "NOT c1._2.length==c2._2.length")
              val sm0 = new Array[Float](c1._1.length)
              for (i <- 0 until c1._1.length) // add 2 arrays
                sm0(i) = c1._1(i) + c2._1(i)
              val sm1 = new Array[Float](c1._1.length)
              for (i <- 0 until c1._2.length) // add 2 arrays
                sm1(i) = c1._2(i) + c2._2(i)
              (sm0, sm1)
            })
        println("treeAggregate time="+(currentTime-time2)/1000.0)

        val debug = false // check if first parameter was correctly aggregated
        if (debug) {
          val sRDD = synRDD.map(x => (x._1, x._2(0))).collect
          var ss0 = 0.0f
          var ss1 = 0.0f
          for ((id, v) <- sRDD) {
            if (id == 0) ss0 += v
            if (id == 1) ss1 += v
          }
          println("sum of first parameter")
          println("  testAvg ss0=" + ss0 + "   testAvg ss1=" + ss1)
          println("   syn0Avg(0)=" + syn0Avg(0) + "    syn1Avg(0)=" + syn1Avg(0))
          println("syn0(0)(0)(0)=" + syn0(0)(0)(0) + " syn1(0)(0)(0)=" + syn1(0)(0)(0))
          require(math.abs(syn0Avg(0) - ss0) < 1E-5, "NOT syn0Avg(0)==ss0")
          require(math.abs(syn1Avg(0) - ss1) < 1E-5, "NOT syn1Avg(0)==ss1")
        }
        /*----------------------------------------------------------------------- */
        /*            weight decay and limit length of embeddings                 */
        /*----------------------------------------------------------------------- */

        var stWgd = stIter
        if (mc.weightDecay > 0.0) {
          require(mc.weightDecay < 0.5, "NOT mc.weightDecay<0.5")
          val numSentenceRdd = trainSet(indexRDD).count()
          val factor = 1 - mc.weightDecay * numSentenceRdd / numSentenceTrainSet
          stWgd += "weightDecayPerRdd=" + Ut.pp(factor)
          for (i <- 0 until syn0Avg.length)
            syn0Avg(i) = factor * syn0Avg(i)
          for (i <- 0 until syn1Avg.length)
            syn1Avg(i) = factor * syn1Avg(i)
        }
        syn0 = mc.unstack(syn0Avg, 0, numPartitions.toFloat)
        syn1 = mc.unstack(syn1Avg, 0, numPartitions.toFloat)
        val numLimit = mc.limitEmbeddingLength(syn0, mc.maxEmbNorm) + mc.limitEmbeddingLength(syn1, mc.maxEmbNorm)
        //stWgd += " numLimit=" + numLimit + " learnRate=" + Ut.pp(currentLearnRate, "%10.6f")
        val iterTime = (System.currentTimeMillis() - iterStart) / 1000
        println(stWgd + " iterSecs=" + Ut.pp(iterTime, "%10.1f"))
        totalWordCount += wordCountTrainAcc.value

      }


      //adjust sense assignment and calculate loss for validation set
      //println("iteration = " + it + "   indexRDD = " + indexRDD + " adjust sense assignment and calculate loss for " +
      //  "validation set...")
      if ((!saveRDD)&&((it+1) == numIterations || (it+1) % mc.modelValidateIter == 0)) {
        /*----------------------------------------------------------------------- */
        /*            adjust senses of validation set and compute loss            */
        /*----------------------------------------------------------------------- */
        val scVali = validationSet.context
        val mcBc = scVali.broadcast(mc)
        val lossValiAcc = scVali.accumulator(0.0)
        val lossNumValiAcc = scVali.accumulator(0)
        val numAdjustValiAcc = scVali.accumulator(0l)
        val syn0Bc = scVali.broadcast(syn0)
        val syn1Bc = scVali.broadcast(syn1)

        validationSet = validationSet.mapPartitionsWithIndex { (idx, iterator) =>
          // always the same seeds for a partition
          val F = new ModelUpdater(mcBc.value, 578829 + idx, syn0Bc.value, syn1Bc.value)

          //          val F = new ModelConst(window, vectorSize, maxNumSenses, numNegative, vocabSize, learningRate,null, null,
          //            expTable, numberOfSensesPerWord, smoothedFrequencyLookupTable, syn0, syn1)
          val newIter = mutable.MutableList[(Array[Int], Array[Array[Int]])]()
          var lossVali = 0.0
          var lossNumVali = 0
          var numAdjust = 0
          for ((sentence, sentenceNEG) <- iterator) {
            if (!F.m.oneSense) {
              var nadj = 0
              while (!F.m.oneSense && nadj < F.m.maxAdjusting) {
                val adjusted = F.adjustSentence(sentence, sentenceNEG) // change the word-senses of sent
                if (adjusted) {
                  nadj += 1
                  numAdjust += 1
                } else
                  nadj = F.m.maxAdjusting
              }
            }
            val (sentLoss, sentLossNum) = F.sentenceLoss(sentence, sentenceNEG)
            lossVali += sentLoss
            lossNumVali += sentLossNum
            newIter += sentence -> sentenceNEG
          }
          lossValiAcc += lossVali
          lossNumValiAcc += lossNumVali;
          numAdjustValiAcc += numAdjust
          newIter.toIterator
        }.cache()
        val sentenceNumValidation = validationSet.count()
        val valiLossPerPredict = lossValiAcc.value / lossNumValiAcc.value
        val valiNumAdjustPerSentence = numAdjustValiAcc.value * 1.0 / sentenceNumValidation
        var st = stIter + "VALISET: lossPerPredict\t=" + Ut.pp(valiLossPerPredict, "%10.6f")
        st += " numPredict=" + lossNumValiAcc.value
        if (!mc.oneSense) st += " numAdjustPerSentence=" + Ut.pp(valiNumAdjustPerSentence, "%10.6f")
        if (mc.printLv > 0)
          println("--------------------------------------------------------------------------------------------------")
        println(st)
        hist(histPos("valiLossPerPredict")) = valiLossPerPredict
        hist(histPos("valiNumAdjustPerSentence")) = valiNumAdjustPerSentence


        if (lastValiLoss >=0 && valiLossPerPredict > lastValiLoss)
          flag = true;
        lastValiLoss = valiLossPerPredict;

        if (!flag) {
          println(stIter + "saving model ...")
          writeToFile(outputPath)

        }
      }

      //println()
      //println("syn0(0)(0)(0)=" + syn0(0)(0)(0))
      //println("syn0Modify(0)(0)=" + syn0Modify(0)(0))
      //println("syn0Modify(0)(0)=" + syn0Modify(0)(0))

      historyWriter.write(hist.mkString(" ") + "\n")
      historyWriter.flush()
      it+=1
    }

    //for (r <- 0 until numRDDs)
    //  trainSet(r).saveAsTextFile(outputPath+"/trainRDD_"+r)
    historyWriter.close()
  }

  /**
    * write to file
    * wordIndex.txt : wordString_senNo
    * vectors.txt:
    *
    * @param outputPath
    */
  private def writeToFile(outputPath: String): Unit = {
    SenseAssignment.backupFile(outputPath + "/wordIndex.txt")
    SenseAssignment.backupFile(outputPath + "/vectors.txt")

    val wordIndexFile = new PrintWriter(new File(outputPath + "/wordIndex.txt"))
    val syn0File = new PrintWriter(new File(outputPath + "/syn0.txt"))
    val syn1File = new PrintWriter(new File(outputPath + "/syn1.txt"))
    val wordStringSyn0File = new PrintWriter(new File(outputPath + "/vectors.txt"))
    val wordIndex = vocabHash.toArray.sortWith((a, b) => a._2 < b._2)

    for ((wordString, iword) <- wordIndex) {
      wordIndexFile.write(wordString + "_" + iword + "_" + mc.numberOfSensesPerWord(iword) + "\n")
    }
    wordIndexFile.close()

    writeSyn(syn0, outputPath + "/syn0.txt")
    writeSyn(syn1, outputPath + "/syn1.txt")

        wordStringSyn0File.write(mc.vocabSize + " " + mc.vectorSize + "\n")
        for ((wordString, iword) <- wordIndex) {
          wordStringSyn0File.write(wordString)
          for (sense <- 0 until mc.numberOfSensesPerWord(iword)) {
            //println(wordString + "_" + sense + " " + word)
            for (i <- 0 until mc.vectorSize )
              wordStringSyn0File.write(" " + syn0(iword)(sense)(i))
            wordStringSyn0File.write("\n")
          }
        }
        wordStringSyn0File.close()
  }

  def writeSyn(syn: Array[Array[Array[Float]]], synFileName: String) {
    SenseAssignment.backupFile(synFileName)
    val synFile = new PrintWriter(new File(synFileName))
    for (iword <- 0 until mc.vocabSize) {
      val word = vocab(iword).word
      for (isense <- 0 until mc.numberOfSensesPerWord(iword)) {
        synFile.write(word + " " + isense + " ")
        for (i <- 0 to mc.vectorSize - 1) {
          synFile.write(syn(iword)(isense)(i) + " ")
        }
        synFile.write("\n")
      }
    }
    synFile.close()
  }

  /**
    * check derivatives with finite differences
    *
    * @param textRDD
    */
  def checkDerivSigmoid(textRDD: RDD[String]) {
    learnVocab(textRDD) // generates vocabulary
    val rnd = new scala.util.Random(seed)

    val (vocabSize, numberOfSensesPerWord) = createNumberOfSensePerWord(rnd) // senseTable has number of senses for each
    // word
    mc.setDictInfo(vocabSize, numberOfSensesPerWord, this.createMultinomFreqDistr())

    val numericRDD: RDD[Array[Int]] = makeSentences(textRDD, rnd) // transform data to numeric
    val tseed = 37

    initSynLocalRandomly(rnd) // init embeddings

    /*------- check conversion to and from BDV ---------*/
    val prmVec0 = mc.syn2bdv(syn0, syn1)
    val (syy0, syy1) = mc.bdv2syn(prmVec0)
    println("check conversion to and from BDV = BreezeDenseVector")
    for (iw <- 0 until syy0.length) {
      for (is <- 0 until syy0(iw).length) {
        for (i <- 0 until syy0(iw)(is).length) {
          require(syy0(iw)(is)(i) == syn0(iw)(is)(i))
          require(syy1(iw)(is)(i) == syn1(iw)(is)(i))
        }
      }
    }

    bdv2synLocal(prmVec0) // initialize syn0 and syn1
    val sent0 = numericRDD.first() // select sentence for testing
    for (pos <- 0 until math.min(sent0.length, 3)) {
      val inpWs = sent0(pos)
      val rnd0 = new Random(tseed)
      val sent0NEG = mc.generateSentenceNEG(sent0, rnd0)
      val modelUpd = new ModelUpdater(mc, seed, syn0, syn1)

      // compute loss and derivative by analytical procedure
      val (lossExact, derivExact) = modelUpd.getLossDerivSigmoid(inpWs, pos, sent0, sent0NEG, true)
      println("SIGMOID: lossExact=" + lossExact)

      val (syz0, syz1) = mc.bdv2syn(prmVec0) // initialize sy0 and sy1 from prmVec
      val modelUpd1 = new ModelUpdater(mc, seed, syz0, syz1)
      val (lossApprox, derivApprox) = modelUpd1.getLossDerivSigmoid(inpWs, pos, sent0, sent0NEG, false)
      val lossDiff = math.abs(lossExact - lossApprox)
      println("SIGMOID: lossApprox=" + lossApprox + " difference to exact value=" + lossDiff)
      require(lossDiff < 0.001, "NOT lossDiff<0.001")

      // compute maximum difference between analytical and approximate derivative
      val parmNames = mc.bdv2names
      var maxdiffApprox = 0.0
      println("SIGMOID: name     prmVec       analyticalDeriv  approxDeriv     (analyticalDeriv-approxDeriv) ")
      for (i <- 0 until derivExact.length) {
        val dif = math.abs(derivExact(i) - derivApprox(i))
        println(parmNames(i) + " " + Ut.pp(prmVec0(i), "%10.6f") + "\t " + Ut.pp(derivExact(i), "%10.6f") + "\t "
          + Ut.pp(derivApprox(i), "%10.6f") + "\t " + Ut.pp(dif, "%10.6f"))
        maxdiffApprox = math.max(maxdiffApprox, dif)
      }
      println("\n----- SIGMOID: maxdiff=" + maxdiffApprox + " between analytical and approximate derivative ------")
      println()
      require(maxdiffApprox < 0.1)

      //---------------- check if analytical derivative is correct ------------
      var func = (prmVec: BDV[Double]) => {
        val (sy0, sy1) = mc.bdv2syn(prmVec) // initialize sy0 and sy1 from prmVec
        val rnd = new Random(tseed)
        val modelUpd1 = new ModelUpdater(mc, seed, sy0, sy1)
        val (loss, deriv): (Double, Array[Float]) = modelUpd1.getLossDerivSigmoid(inpWs, pos, sent0, sent0NEG, true)
        loss
      }
      val h = 0.001
      val (finiteDiffApprox, gradComputeErr) = EmpiricalDerivative(func, prmVec0, h)

      // compute maximum difference between analytical and finite difference approximation
      var maxdiff = 0.0
      println("SIGMOID: name     prmVec       analytical  finiteDiff     (derivExact-finitDiff) ")
      for (i <- 0 until derivExact.length) {
        val dif = math.abs(derivExact(i) - derivApprox(i))
        println(parmNames(i) + " " + Ut.pp(prmVec0(i), "%10.6f") + "\t " + Ut.pp(derivExact(i), "%10.6f") + "\t "
          + Ut.pp(finiteDiffApprox(i), "%10.6f") + "\t " + Ut.pp(dif, "%10.6f"))
        maxdiff = math.max(maxdiff, dif)
      }
      println("\n----- SIGMOID: maxdiff=" + maxdiff + " with exact function"
        + " between analytical deriv.and finite difference approximation ------")
      println()
      require(maxdiff < 0.1)

      // check if subtracting derivative reduces loss
      val prmVec2: BDV[Double] = prmVec0.map(x => x)
      for (i <- 0 until prmVec2.length)
        prmVec2(i) += -0.02 * derivExact(i)
      val (sy0, sy1) = mc.bdv2syn(prmVec2) // initialize sy0 and sy1 from prmVec
      val modelUpd2 = new ModelUpdater(mc, seed, sy0, sy1)
      val (loss2, deriv2): (Double, Array[Float]) = modelUpd2.getLossDerivSigmoid(inpWs, pos, sent0, sent0NEG, true)
      println("SIGMOID: loss0=" + lossExact + " before. loss2=" + loss2 + " after subtracting 0.02*deriv")
      println(); require(loss2 < lossExact, "NOT loss2<loss0")

      val alpha = 0.1f

      //      val (lossApprox, derivApprox) = modelUpd.getLossDerivSigmoid(inpWs, pos, sent0, sent0NEG, false)
      //      val (lossApprox1, derivApprox1) = modelUpd.getLoss(inpWs, pos, sent0, sent0NEG)
      //      require(lossApprox1 == lossApprox, "getLoss: wrong loss")

      val (loss00, n00) = modelUpd.learnWordSense(inpWs, pos, 0.0f, sent0, sent0NEG) //loss: no change of param
      require(loss00 == lossApprox, loss00 + "=loss00!=lossApprox=" + lossApprox)
      val n = 10
      val lossArr = new Array[Double](n)
      for (i <- 0 until n) {
        val (lossA, nA) = modelUpd.learnWordSense(inpWs, pos, alpha, sent0, sent0NEG)
        lossArr(i) = lossA
      }
      println("SIGMOID: lossExact \t=" + lossExact)
      println("SIGMOID: lossApprox\t=" + lossApprox)
      println("SIGMOID: loss00    \t=" + loss00 + "\t computed by learnWordSense")
      for (i <- 0 until n) {
        println(" loss(" + i + ")\t=" + lossArr(i) + "\t changing parameters by learnWordSense")
        if (i > 0) require(lossArr(i - 1) > lossArr(i))
      }
      println("SIGMOID: finished pos=" + pos)
    }
  }


  /**
    * check derivatives with finite differences
    *
    * @param textRDD
    */
  def checkDerivSoftmax(textRDD: RDD[String]) {
    learnVocab(textRDD) // generates vocabulary
    val rnd = new scala.util.Random(seed)

    val (vocabSize, numberOfSensesPerWord) = createNumberOfSensePerWord(rnd) // senseTable has number of senses for each
    // word
    mc.setDictInfo(vocabSize, numberOfSensesPerWord, this.createMultinomFreqDistr())

    val numericRDD: RDD[Array[Int]] = makeSentences(textRDD, rnd) // transform data to numeric
    val tseed = 37
    val syy0 = mc.initSynRand(rnd)
    val syy1 = mc.initSynRand(rnd)
    val prmVec0 = mc.syn2bdv(syy0, syy1)

    val prmVeca = prmVec0.map(x => 10.0 * x)
    bdv2synLocal(prmVeca) // initialize syn0 and syn1

    //--------------- check writing params to file

    val folder: File = File.createTempFile("tempDir", null);
    folder.delete()
    folder.mkdir() // create temporary directory

    val syn0Old = mc.synClone(syn0)
    val syn1Old = mc.synClone(syn1)
    writeToFile(folder.getCanonicalPath)

    initSynFromFile(folder.getCanonicalPath, rnd)

    mc.synEqual(syn0Old, syn0)
    mc.synEqual(syn1Old, syn1)


    //--------------- check derivatives

    val sent0 = numericRDD.first()
    for (pos <- 0 until math.min(3, sent0.length)) {
      val inpWs = sent0(pos)
      val rnd0 = new Random(tseed)
      val sent0NEG = mc.generateSentenceNEG(sent0, rnd0)
      val modelUpd = new ModelUpdater(mc, seed, syn0, syn1)
      val (lossExact, derivExact): (Double, Array[Float]) = modelUpd.getLossDerivLogSoftmax(inpWs, pos, sent0, sent0NEG,
        true)
      println("SOFTMAX: lossExact=" + lossExact)

      for (exact <- Array(true, false)) {
        {
          //--------------- check inner derivative
          val z = BDV.fill[Double](4)(rnd.nextDouble() - 0.5)
          //val zmx = Bmax(z)
          //for(i<- 0 until z.length) z(i)-zmx
          val x = z(0)
          val y = z(1 until z.length).toArray
          val (loss2, derX, derY) = modelUpd.logSoftmax(x, y, true)

          var fct = (zz: BDV[Double]) => {
            val xx = zz(0)
            val yy = zz(1 until zz.length).toArray
            val (loss, derivx, derivy) = modelUpd.logSoftmax(xx, yy, true)
            loss
          }

          val h = 0.0001
          val (grdVec, grdComputeErr) = EmpiricalDerivative(fct, z, h);
          println("SOFTMAX: ")
          println(derX + "\t " + grdVec(0) + "\t " + (derX - grdVec(0)));
          var mxDiff = math.abs(derX - grdVec(0))
          for (i <- 0 until derY.length) {
            println(derY(i) + "\t " + grdVec(i + 1) + "\t " + (derY(i) - grdVec(i + 1)));
            mxDiff = math.max(mxDiff, math.abs(derX - grdVec(0)))
          }
          println("SOFTMAX: inner derivative mxDiff=" + mxDiff)
        }
        {
          //--------------- check full derivative
          var func = (prmVec: BDV[Double]) => {
            val (sy0, sy1) = mc.bdv2syn(prmVec) // initialize sy0 and sy1 from prmVec
            val rnd = new Random(tseed)
            val modelUpd1 = new ModelUpdater(mc, seed, sy0, sy1)
            val (loss, deriv): (Double, Array[Float]) = modelUpd1
              .getLossDerivLogSoftmax(inpWs, pos, sent0, sent0NEG, exact)
            loss
          }

          val h = 0.0001
          val (gradVec, gradComputeErr) = EmpiricalDerivative(func, prmVeca, h)
          //val (gradVec1, gradComputeErr1) = EmpDeriv.adaptiveStepsizeVec(func, initialWeights1, h, Nil)
          var maxdiff = 0.0
          val parmNames = mc.bdv2names
          println("SOFTMAX:  name     prmVec  program   finiteDiff  difference with exact=" + exact)
          for (i <- 0 until derivExact.length) {
            println(parmNames(i) + " " + prmVeca(i) + "\t " + derivExact(i) + "\t " + gradVec(i) + "\t " +
              (derivExact(i) - gradVec(i)))
            maxdiff = math.max(maxdiff, math.abs(derivExact(i) - gradVec(i)))
          }
          println("\n--------------------- SOFTMAX: full deriv maxdiff=" + maxdiff + " with exact=" + exact +
            "--------------------------")
          println()
          require(maxdiff < 0.1 || !exact, "NOT maxdiff<0.1" + " with exact=" + exact)
          // initialize sy0 and sy1 from prmVec
          val (syz0, syz1) = mc.bdv2syn(prmVeca)
          val modelUpd1 = new ModelUpdater(mc, seed, syz0, syz1)
          val (loss1, deriv1): (Double, Array[Float]) = modelUpd1
            .getLossDerivLogSoftmax(inpWs, pos, sent0, sent0NEG, exact)

          val prmVec2: BDV[Double] = prmVeca.map(x => x)
          for (i <- 0 until prmVec2.length)
            prmVec2(i) += -0.02 * deriv1(i)
          // initialize sy0 and sy1 from prmVec
          val (sy0, sy1) = mc.bdv2syn(prmVec2)
          val modelUpd2 = new ModelUpdater(mc, seed, sy0, sy1)
          val (loss2, deriv2): (Double, Array[Float]) = modelUpd2
            .getLossDerivLogSoftmax(inpWs, pos, sent0, sent0NEG, exact)
          println("SOFTMAX: loss0=" + lossExact + " before. loss2=" + loss2 + " after subtracting 0.02*deriv with " +
            "exact=" + exact)
          require(lossExact > loss2, "NOT lossExact >loss2")
        }
        val alpha = 0.001f

        val (lossApprox, derivApprox) = modelUpd.getLossDerivLogSoftmax(inpWs, pos, sent0, sent0NEG, false)
        val (lossApprox1, derivApprox1) = modelUpd.getLoss(inpWs, pos, sent0, sent0NEG)
        require(lossApprox1 == lossApprox, "getLoss: wrong loss")

        val (loss00, n00) = modelUpd.learnWordSense(inpWs, pos, 0.0f, sent0, sent0NEG)
        require(loss00 == lossApprox, loss00 + "=loss00!=lossApprox=" + lossApprox)
        val n = 10
        val lossArr = new Array[Double](n)
        for (i <- 0 until n) {
          val (lossA, nA) = modelUpd.learnWordSense(inpWs, pos, alpha, sent0, sent0NEG)
          lossArr(i) = lossA
        }
        println(" lossExact \t=" + lossExact)
        println(" lossApprox\t=" + lossApprox)
        println(" loss00    \t=" + loss00 + "\t computed by learnWordSense")
        for (i <- 0 until n) {
          println(" loss(" + i + ")\t=" + lossArr(i) + "\t changing parameters by learnWordSense")
          if (i > 0) require(lossArr(i - 1) > lossArr(i)) // require falling loss
        }
        println("SOFTMAX: finished pos=" + pos)
      }
    }
  }

}

object SenseAssignment {

  def readSyn(vocabHash: mutable.HashMap[String, Int], synFileName: String, rand: Random, standardDev: Float
  = 0.01f, numberOfSensesPerWord: Array[Int]): Array[Array[Array[Float]]] = {
    val vocabSize = vocabHash.size
    val lines0 = Source.fromFile(synFileName).getLines.map(line => line.split(" ")).toArray
    val vectorSize = lines0(0).length - 2
    val wordInitialized = new Array[Boolean](vocabSize)
    val syn = ModelConst.initSynZero(numberOfSensesPerWord, vectorSize)
    for (lin <- lines0) {
      val word = lin(0)
      val iword = vocabHash.getOrElse(word, -1)
      require(iword >= 0, "word " + word + " not found")
      val isense = lin(1).toInt
      require(isense < numberOfSensesPerWord(iword), "NOT isense< mc.numberOfSensesPerWord(iword)")
      require(lin.length == vectorSize + 2, "NOT lin.length== mc.vectorSize+2 ")
      if (!wordInitialized(iword)) {
        for (isen <- 0 until numberOfSensesPerWord(iword))
          for (j <- 0 until vectorSize)
            syn(iword)(isen)(j) = lin(j + 2).toFloat + (rand.nextFloat() - 0.5f) * standardDev / vectorSize
      } else {
        for (j <- 0 until vectorSize)
          syn(iword)(isense)(j) = lin(j + 2).toFloat
      }
      wordInitialized(iword) = true
    }
    syn
  }


  def readSynToVector(synFileName: String): (Map[String, Int], Map[String, Int], Array[Float], Int) = {
    val word2numSense = new mutable.HashMap[String, Int]()
    val wordSense2ind = new mutable.HashMap[String, Int]()
    val lines0 = Source.fromFile(synFileName).getLines.map(line => line.split(" ")).toArray
    val vectorSize = lines0(0).length - 2
    val senseVec = new Array[Float](lines0.length * vectorSize)
    var ipos = 0
    for (lin <- lines0) {
      val word = lin(0)
      val sense = lin(1).toInt
      val maxSense = math.max(sense + 1, word2numSense.getOrElse(word, -1))
      word2numSense.put(word, maxSense)
      wordSense2ind.put(word + "_" + sense, ipos)
      val ioff = ipos * vectorSize
      for (i <- 0 until vectorSize)
        senseVec(ioff + i) = lin(2 + i).toFloat
      ipos += 1
    }
    require(ipos * vectorSize == senseVec.length, "NOT ipos*vectorSize == senseVec.length")
    (word2numSense.toMap, wordSense2ind.toMap, senseVec, vectorSize)
  }


  def loadModelSenses(path: String): Word2VecModel = {
    val wordIndex = collection.mutable.Map[String, Int]()
    var index = 0
    for (word <- Source.fromFile(path + "/wordIndex.txt").getLines()) {
      wordIndex.put(word, index)
      index += 1
    }
    val wordVectors = Source.fromFile(path + "/syn0.txt").getLines().map(line => line.split(" ").toSeq).flatten
      .map(s => s.toFloat).toArray
    new Word2VecModel(wordIndex.toMap, wordVectors)
  }

  //  def loadEmbeddings(path: String): Word2VecModel = {
  //    val vocabHash = collection.mutable.HashMap[String, Int]()
  //    val wordSense = collection.mutable.HashMap[Int, Int]()
  //    val wordSenseIndex = collection.mutable.Map[String, Int]()
  //    var maxIndex = 0
  //    var iws = 0
  //    for (line <- Source.fromFile(path + "/wordIndex.txt").getLines()) {
  //      val parts = line.split("_")
  //      vocabHash.put(parts(0), parts(1).toInt)
  //      maxIndex = math.max(maxIndex, parts(1).toInt)
  //      wordSense.put(parts(1).toInt, parts(2).toInt)
  //      for (isense <- 0 until parts(2).toInt) {
  //        wordSenseIndex.put(parts(0) + "_" + parts(1), iws)
  //        iws += 1
  //      }
  //
  //    }
  //    require(maxIndex + 1 == vocabHash.size, "NOT maxIndex== wordIndex.size")
  //    val numberOfSensesPerWord = new Array[Int](vocabHash.size)
  //    for (iw <- 0 until numberOfSensesPerWord.length) {
  //      val ns = wordSense.getOrElse(iw, -1)
  //      require(ns > 0, "NOT ns>0")
  //      numberOfSensesPerWord(iw) = ns
  //    }
  //    val syn0 = SenseAssignment.readSyn(vocabHash, path + "/syn0.txt", new Random(), 0.0f, numberOfSensesPerWord)
  //    //val wordVectors = Source.fromFile(path + "/syn0.txt").getLines().map(line => line.split(" ").toSeq).flatten
  //    //  .map(s => s.toFloat).toArray
  //    //require(iws = wordVectors.length)
  //    new Word2VecModel(vocabHash.toMap, wordVectors)
  //  }


  def backupFile(fileName: String, suffix: String = ".bak"): Unit = {
    val bakFileName = fileName + suffix
    val fibak = new File(bakFileName)
    if (fibak.exists()) {
      //println("deleting file " + bakFileName)
      removeFileOrDirectory(fibak)
    }
    val fibak1 = new File(bakFileName)
    require(!fibak1.exists(), "fibak1.exists()")

    val fi = new File(fileName)
    if (fi.exists()) {
      //println("moving file " +fileName +" to"+ bakFileName)
      fi.renameTo(fibak1)
    }
    val fi1 = new File(fileName)
    require(!fi1.exists(), "fi1.exists()")

  }

  def removeFileOrDirectory(fil: File) {
    if (fil.isDirectory()) {
      val files = fil.listFiles();
      if (files != null && files.length > 0) {
        for (aFile <- files) {
          removeFileOrDirectory(aFile);
        }
      }
      fil.delete();
    } else {
      fil.delete();
    }
  }

}


/**
  * multinomial random number generator by tree lookup
  *
  * @param probabilities unscale probabilities
  */
class UtMultinomial(val probabilities: Array[Double], var rand: Random) extends Serializable {
  val nprob = probabilities.length
  val range = nprob + 1;
  // We build the distribution array one larger than the array of probabilities
  // to permit distribution[0] to act as a minimum bound for searching.
  // Otherwise each distribution value is a maximum bound.
  val distribution = new Array[Double](range);
  var sumProb = 0.0
  for (prob <- probabilities) {
    sumProb += prob
  }
  distribution(0) = 0.0
  for (i <- 1 until range) {
    distribution(i) = distribution(i - 1) + (probabilities(i - 1) / sumProb)
  }
  distribution(range - 1) = 1.0

  def setRandom(rnd: Random): Unit = {
    rand = rnd
  }

  def sample(): Int = {
    // Straightforward binary search on an array of doubles to find
    // index such that distribution[i] is greater than random number while
    // distribution[i-1] is less.
    val key = rand.nextDouble()
    var mindex = 1
    var maxdex = range - 1
    var midpoint = mindex + (maxdex - mindex) / 2
    while (mindex <= maxdex) {
      // System.out.println(midpoint);
      if (key < distribution(midpoint - 1)) {
        // This shouldn't ever produce an out of bounds error, since it's impossible
        // that the key will be less than 0, and thus impossible that the midpoint will ever be
        // zero.  I think.
        maxdex = midpoint - 1
      } else if (key > distribution(midpoint)) {
        mindex = midpoint + 1
      } else {
        return midpoint - 1
        // minus one, because the whole distribution array is shifted one up from the
        // original probabilities array to permit distribution[0] to be a minbound.
      }
      midpoint = mindex + math.ceil((maxdex - mindex) / 2).toInt
      // I use Math.ceil to avoid any possibility of midpoint = 0.
    }
    throw new RuntimeException("Error in multinomial sampling method.")
    return range - 1;
  }


}

@SerialVersionUID(1000L)
object CountArrAccumulatorParam extends AccumulatorParam[Array[Array[Int]]] {
  //  override def zero(numSense:Array[Int]): Array[Array[Int]] = {
  //    val senseCount = new Array[Array[Int]](numSense.length) // an array of Int for the senses of each word
  //    for (w <- 0 until numSense.length) {
  //      senseCount(w) = new Array[Int](numSense(w)) // length of array = number of senses
  //    }
  //    senseCount
  //  }

  override def zero(initialValue: Array[Array[Int]]): Array[Array[Int]] = {
    val senseCount = new Array[Array[Int]](initialValue.length) // an array of Int for the senses of each word
    for (w <- 0 until initialValue.length) {
      senseCount(w) = new Array[Int](initialValue(w).length) // length of array = number of senses
    }
    senseCount
  }

  override def addInPlace(v1: Array[Array[Int]], v2: Array[Array[Int]]): Array[Array[Int]] = {
    require(v1.length == v2.length)
    for (iw <- 0 until v1.length)
      for (is <- 0 until (v1(iw).length))
        v1(iw)(is) += v2(iw)(is)
    v1
  }
}
