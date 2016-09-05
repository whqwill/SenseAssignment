package de.fraunhofer.iais.kd.haiqing

import java.io.{File, PrintWriter}

import com.github.fommil.netlib.BLAS._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import com.github.fommil.netlib.BLAS.{getInstance => blas}

import scala.collection.mutable

/**
  * Created by hwang on 26.04.16.
  */

case class VocabWord(
                      var word: String,
                      var cn: Int
                    )

/**
  * model for computing nearest neighbors
  *
  * @param wordIndex word-sense string to running number
  * @param wordVectors all embeddings in a single array
  */
class Word2VecModel(
                     private val wordIndex: Map[String, Int],
                     private val wordVectors: Array[Float]) {


  private val numWords = wordIndex.size
  private val vectorSize = wordVectors.length / numWords

  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }

  private val wordVecNorms: Array[Double] = {
    val wordVecNorms = new Array[Double](numWords)
    var i = 0
    while (i < numWords) {
      val vec = wordVectors.slice(i * vectorSize, i * vectorSize + vectorSize)
      wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)  // sqrt(vec'*vec)
      i += 1
    }
    wordVecNorms
  }

  @inline final def getEmb(iws:Int):Array[Float]={
    val vec = wordVectors.slice(iws * vectorSize, iws * vectorSize + vectorSize)
    vec
  }

  private def cosineSimilarity(ws1: String, ws2: String): Double = {
    cosineSimilarity(getEmb(ws1), getEmb(ws2))
  }

  private def cosineSimilarity(iws1: Int, iws2: Int): Double = {
    cosineSimilarity(getEmb(iws1), getEmb(iws2))
  }

  private def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.snrm2(n, v1, 1)  // sqrt(v1'*v1)
    val norm2 = blas.snrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.sdot(n, v1, 1, v2, 1) / (norm1 * norm2)
  }

  def getEmb(word: String): Array[Float] = {
    wordIndex.get(word) match {
      case Some(iws) =>
        getEmb(iws:Int)
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  def word2embeddingVector(word: String, toNorm1:Boolean=true): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors.slice(ind * vectorSize, ind * vectorSize + vectorSize)
        val factor = this.wordVecNorms(ind)
        Vectors.dense(vec.map(x => (x/factor).toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  def findSynonyms(word: String, num: Int,cosineDist:Boolean): Array[(String, Double)] = {
    val vector = word2embeddingVector(word,cosineDist)
    findSynonyms(vector, num,cosineDist)
  }

  def findSynonyms(vector: Vector, num: Int,cosineDist:Boolean): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")
    // TODO: optimize top-k
    val fVector = vector.toArray.map(_.toFloat)
    val cosineVec = Array.fill[Float](numWords)(0)
    val alpha: Float = 1
    val beta: Float = 0

    //    SUBROUTINE SGEMV ( TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY )
    //    #  SGEMV  performs one of the matrix-vector operations
    //    #     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
    //    #  where alpha and beta are scalars, x and y are vectors and A is an m by n matrix.
    //    #              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
    //    #              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
    //    #              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.
    // cosineVec = A' * vector
    blas.sgemv(
      "T", vectorSize, numWords, alpha, wordVectors, vectorSize, fVector, 1, beta, cosineVec, 1)

    // Need not divide with the norm of the given vector since it is constant.
    val cosVec = cosineVec.map(_.toDouble)
    if(cosineDist) {
      var ind = 0
      while (ind < numWords) {
        cosVec(ind) /= wordVecNorms(ind)
        ind += 1
      }
    }

    wordList.zip(cosVec)
      .toSeq
      .sortBy(-_._2)
      .take(num + 1)
      //.tail
      .toArray
  }

  def getVectors: Map[String, Array[Float]] = {
    wordIndex.map { case (word, ind) =>
      (word, wordVectors.slice(vectorSize * ind, vectorSize * ind + vectorSize))
    }
  }

  def save(path: String): Unit = {
    val file1 = new PrintWriter(new File(path + "/wordIndex.txt"))
    val file2 = new PrintWriter(new File(path + "/wordVectors.txt"))
    val iter = wordIndex.toIterator
    while (iter.hasNext) {
      val tmp = iter.next()
      file1.write(tmp._1 + " " + tmp._2 + "\n")
    }
    for (i <- 0 to wordVectors.size - 2)
      file2.write(wordVectors(i) + " ")
    file2.write(wordVectors(wordVectors.size - 1) + "\n")
    file1.close()
    file2.close()
  }

  def avgSim(word1 : String, word2 : String, word2numSense:Map[String,Int]): Double = {

    val wsArr1 = {
      val nsense = word2numSense.getOrElse(word1, -1)
      require(nsense > 0, "word " + word1 + " not in word2numSense")
      (0 until nsense).map(isense => word1 + "_" + isense).toArray
    }
    val wsArr2 = {
      val nsense = word2numSense.getOrElse(word2, -1)
      require(nsense > 0, "word " + word2 + " not in word2numSense")
      (0 until nsense).map(isense => word2 + "_" + isense).toArray
    }

    var sum = 0.0
    for (w1 <- wsArr1)
      for (w2 <- wsArr2)
        sum += cosineSimilarity(w1,w2)
    sum/wsArr1.length/wsArr2.length
  }

  def maxSim(word1 : String, word2 : String, word2numSense:Map[String,Int]): Double = {

    val wsArr1 = {
      val nsense = word2numSense.getOrElse(word1, -1)
      require(nsense > 0, "word " + word1 + " not in word2numSense")
      (0 until nsense).map(isense => word1 + "_" + isense).toArray
    }
    val wsArr2 = {
      val nsense = word2numSense.getOrElse(word2, -1)
      require(nsense > 0, "word " + word2 + " not in word2numSense")
      (0 until nsense).map(isense => word2 + "_" + isense).toArray
    }

    var ans = -1.0
    for (w1 <- wsArr1)
      for (w2 <- wsArr2)
        if (cosineSimilarity(w1,w2) > ans)
          ans = cosineSimilarity(w1,w2)
    ans
  }

  def getNeighbors(searchTerms:Array[String], word2numSense:Map[String,Int], numSynonyms:Int, cosineDist:Boolean) {
    for (j <- 0 until searchTerms.length) {
      val parts = searchTerms(j).split("_")
      val wsArr = parts.length match {
        case 1 => {
          val nsense = word2numSense.getOrElse(searchTerms(j), -1)
          require(nsense > 0, "word " + searchTerms(j) + " not in word2numSense")
          (0 until nsense).map(isense => searchTerms(j) + "_" + isense).toArray
        }
        case 2 => Array(searchTerms(j))
        case _ => throw new RuntimeException("wrong search entry " + searchTerms(j))
      }

      if(wsArr.length>1){
        var len=10
        for(i <- 0 until wsArr.length)
          len = math.max(len,wsArr(i).length)
        val fmt = "%"+len+".6f"
        var st0 = "".padTo(len,' ') + " "
        for(i <- 0 until wsArr.length)
          st0 += wsArr(i).padTo(len,' ') + " "
        println(st0)
        for(i <- 0 until wsArr.length) {
          var st = wsArr(i).padTo(len,' ') + " "
          for (j <- 0 until wsArr.length) {
            val dist = cosineSimilarity(wsArr(i), wsArr(j))
            st += Ut.pp(dist,fmt)
          }
          println(st)
        }
      }

      // find neighbors of the word-senses of a word
      for (ws <- wsArr) {
        val word = ws.split("_")(0)
        val st0 = if(cosineDist) "cosine " else "Euclidean "
        val st = if(word2numSense.getOrElse(searchTerms(j), -1)==1) word else ws
        println("------  nearest "+st0+"dist neighbors of " + st + "  -----")
        val synonyms = findSynonyms(ws, numSynonyms,cosineDist)
        var mxSize = 0
        for ((synonym, cosineSimilarity) <- synonyms) {
          mxSize = math.max(synonym.size,mxSize)
        }

        for ((synonym, cosineSimilarity) <- synonyms) {
          val syTok=synonym.split("_")(0)
          val sy = if(word2numSense.getOrElse(syTok, -1)==1) syTok else synonym
          //val tok = synonym.substring(0, synonym.size - 2)
          //print((sy.split("_"))(0)+", ")
	  print(sy+", ")
        }
        println("---------------------------------------------------------------------------------------")
      }
      println("=========================================================================================")
      println()
    }
  }

  def saveNeighbors(searchTerms:Array[String], word2numSense:Map[String,Int], numSynonyms:Int, cosineDist:Boolean, pathFolder:String) {
    //val file_vec = new PrintWriter(new File(pathFolder+"some_vec.txt"))
    val file_word = new PrintWriter(new File(pathFolder+"/"+"some_word.txt"))
    val hash = new mutable.HashSet[String]()

    for (j <- 0 until searchTerms.length) {

      val parts = searchTerms(j).split("_")
      val wsArr = parts.length match {
        case 1 => {
          val nsense = word2numSense.getOrElse(searchTerms(j), -1)
          require(nsense > 0, "word " + searchTerms(j) + " not in word2numSense")
          (0 until nsense).map(isense => searchTerms(j) + "_" + isense).toArray
        }
        case 2 => Array(searchTerms(j))
        case _ => throw new RuntimeException("wrong search entry " + searchTerms(j))
      }

      //val fileOneWord_vec = new PrintWriter(new File(pathFolder+wsArr(0).split("_")(0)+"_vec.txt"))
      val fileOneWord_word = new PrintWriter(new File(pathFolder+"/"+wsArr(0).split("_")(0)+"_word.txt"))
      val hashOneWord = new mutable.HashSet[String]()

      println(pathFolder+"/"+wsArr(0).split("_")(0)+"_word.txt")

      // find neighbors of the word-senses of a word
      for (ws <- wsArr) {
        val synonyms = findSynonyms(ws, numSynonyms,cosineDist)
        for (synonym <- synonyms) {
          hashOneWord.add(synonym._1)
          hash.add(synonym._1)
        }
      }

      val oneWord_words = hashOneWord.toList
      for (word <- oneWord_words) {
        fileOneWord_word.write(word+"\n")

        //fileOneWord_vec.write()
      }
      fileOneWord_word.close()
    }

    val words = hash.toList
    for (word <- words)
      file_word.write(word+"\n")
    file_word.close()
  }

  //def avgSim()

}

object TestSenseVectors {
  /**
    * determine nearest neighbors of words
    *
    * @param args first arg is the path of the syn0-file
    *             second arg is number of neighbors
    *             additional arguments are the words or word-senses (word_sense) for which the neighbors are to be
    *             determined
    */
  def main(args: Array[String]): Unit = {
    require(args.length>=3,"args.length<3")
    val synFileName = args(0)+"/syn0.txt"
    val numSynonyms = args(1).toInt
    println("----- determine "+numSynonyms+" closest embeddings from file "+synFileName)
    val (word2numSense, wordSense2ind, senseVec, vectorSize) = SenseAssignment.readSynToVector(synFileName)
    val model = new Word2VecModel(wordSense2ind, senseVec)

    val searchTerms = (2 until args.length).map(i => args(i)).toArray

    val cosineDist = true
    model.getNeighbors(searchTerms, word2numSense, numSynonyms,cosineDist)


//    val NEWsynonyms = model.findSynonyms(args(2), 20)
//    //val synonyms = model.findSynonyms("day", 10)
//
//    for ((synonym, cosineSimilarity) <- NEWsynonyms) {
//      println(s"${synonym.substring(0, synonym.size - 2)} $cosineSimilarity")
//    }
//    println()

  }
}

