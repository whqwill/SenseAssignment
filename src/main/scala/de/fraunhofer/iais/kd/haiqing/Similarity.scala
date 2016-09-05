package de.fraunhofer.iais.kd.haiqing

import java.io.{File, PrintWriter}

import scala.io.Source

/**
 * Created by hwang on 27.06.16.
 */

object Similarity {
  def main(args: Array[String]): Unit = {
    require(args.length>=3,"args.length<3")
    val synFileName = args(0)+"/syn0.txt"
    val (word2numSense, wordSense2ind, senseVec, vectorSize) = SenseAssignment.readSynToVector(synFileName)
    val model = new Word2VecModel(wordSense2ind, senseVec)
    Source.fromFile(args(1)).getLines.toArray

    val lines0 = Source.fromFile(args(1)).getLines.map(line => Tuple2(line.split(",")(0),line.split(",")(1))).toArray
    val file = new PrintWriter(new File(args(2)))
    for (line <- lines0) {
      file.write(model.avgSim(line._1.toLowerCase, line._2.toLowerCase, word2numSense).toString + "\n")
      println(model.avgSim(line._1.toLowerCase, line._2.toLowerCase, word2numSense))
    }
    file.close()
  }
}
