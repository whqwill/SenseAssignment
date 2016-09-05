package de.fraunhofer.iais.kd.haiqing

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics

/**
 * Created by hwang on 27.06.16.
 */
object test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Sense2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
      conf.setMaster("local[4]")
    }
    val sc = new SparkContext(conf)
    val seriesX = sc.textFile(args(0)).map(x=>x.toDouble)
    val seriesY = sc.textFile(args(1)).map(x=>x.toDouble)
    val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson")
    println(correlation)
  }

}
