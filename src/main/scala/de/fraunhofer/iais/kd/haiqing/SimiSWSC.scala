package de.fraunhofer.iais.kd.haiqing

import java.io.{File, PrintWriter}

import com.github.fommil.netlib.BLAS.{getInstance => blas}

import scala.io.Source

/**
  * Created by haiqing on 29.06.16.
  */
object SimiSWSC {
    def main(args: Array[String]): Unit = {
      require(args.length>=3,"args.length<3")
      val syn0FileName = args(0)+"/syn0.txt"
      val syn1FileName = args(0)+"/syn1.txt"
      val (word2numSense0, wordSense2ind0, senseVec0, vectorSize0) = SenseAssignment.readSynToVector(syn0FileName)
      val (word2numSense1, wordSense2ind1, senseVec1, vectorSize1) = SenseAssignment.readSynToVector(syn1FileName)
      val model0 = new Word2VecModel(wordSense2ind0, senseVec0)
      val model1 = new Word2VecModel(wordSense2ind1, senseVec1)
      Source.fromFile(args(1)).getLines.toArray

      val lines = Source.fromFile(args(1)).getLines.map(line => Tuple4(line.split("\t")(0),line.split("\t")(1),line.split("\t")(2),line.split("\t")(3))).toArray
      val file = new PrintWriter(new File(args(2)))
      for (line <- lines) {
        val word1 = line._1.toLowerCase
        val word2 = line._2.toLowerCase
        if (word2numSense0.get(word1).nonEmpty && word2numSense0.get(word2).nonEmpty) {

          val context1 = line._3.split(" ").filter(x => word2numSense1.get(x).nonEmpty)
          val context2 = line._4.split(" ").filter(x => word2numSense1.get(x).nonEmpty)
          var score1 = -1.0f
          var w1 = ""
          for (i <- 0 until word2numSense0.get(word1).get) {
            val word = word1 + "_" + i
            val v0 = model0.getEmb(word)
            var s = 1.0f
            for (u0 <- context1) {
              val v1 = model1.getEmb(u0 + "_0")
              s *= Score(v0, v1)
            }
            if (s > score1) {
              score1 = s
              w1 = word
            }
          }
          println(w1)
          var score2 = -1.0f
          var w2 = ""
          for (i <- 0 until word2numSense0.get(word2).get) {
            val word = word2 + "_" + i
            val v0 = model0.getEmb(word)
            var s = 1.0f
            for (u0 <- context2) {
              val v1 = model1.getEmb(u0 + "_0")
              s *= Score(v0, v1)
            }
            if (s > score2) {
              score2 = s
              w2 = word
            }
          }
          println(w2)

          val sim = cosineSimilarity(model0.getEmb(w1), model0.getEmb(w2))*10
          println(sim)
          file.write(sim.toString + "\n")
          //model0.wordIndex.contains()
          //file.write(model.maxSim(line._1.toLowerCase, line._2.toLowerCase, word2numSense).toString + "\n")
          //println(model.maxSim(line._1.toLowerCase, line._2.toLowerCase, word2numSense))
        }
      }
      file.close()
    }

  def Score(v1: Array[Float], v2: Array[Float]) : Float = {
    actFunSigmoidApprox(v1,v2)
  }

  private def actFunSigmoidApprox(v0: Array[Float], v1: Array[Float]): Float = {
    val vectorSize = v0.length
    // f = v0^T * v1
    val f = blas.sdot(vectorSize, v0, 1, v1, 1)
    val res = 1.0f/(1+math.exp(-f))
    res.toFloat
  }

  private def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.snrm2(n, v1, 1)  // sqrt(v1'*v1)
    val norm2 = blas.snrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.sdot(n, v1, 1, v2, 1) / (norm1 * norm2)
  }

}
