package de.fraunhofer.iais.kd.haiqing

import java.util

import breeze.numerics._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import breeze.linalg.{DenseMatrix => BDM, CSCMatrix => BSM, DenseVector => BDV, SparseVector => BSV, Vector => BV,
Matrix => BM, axpy => Baxpy, sum => Bsum, max => Bmax, min => Bmin, *}
import breeze.numerics.{log => Blog, sigmoid => Bsigmoid, exp => Bexp, abs => Babs, ceil}


import scala.util.Random

/**
  * Created by gpaass on 14.05.16.
  */
object SuiteTest {

  def main(args: Array[String]): Unit = {

    //val sc = TestUtils.getContext()
    doLossTest()
    doSigmoidTest()
    println("\n\n============== FINISHED ==============")
  }


  def doLossTest(): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN) // omit all iteration logging
    Logger.getLogger("akka").setLevel(Level.WARN) // omit all iteration logging
    val conf = new SparkConf().setAppName("Sence2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
      conf.setMaster("local[2]")
    }
    val sc = new SparkContext(conf)
    println("sc.defaultParallelism=" + sc.defaultParallelism + "   " + sc.master)

    val inputFile = ""
    val numRDD = 1
    val numEpoch = 10
    val minCount = 1
    val count2numSenses = "15"
    val negative = 2
    val window = 1
    val vectorSize = 2
    val maxNumSenses = 2
    val seed = 42l
    val learningRate = 0.25f
    val stepSize = 10000
    val local = true
    val softMax = false
    val maxEmbNorm = 5.0f
    val senseProbThresh = 0.1f // re-initialize embedding if senseProbability is lower

    for (oneSense <- Array(false, true)) {
      for (softMax <- Array(false)) {
        for (syn1OneSense <- Array(true, false)) {
          val smod = new SenseAssignment(
            inputFile, // filename of input
            numRDD, // number of RDDs
            numEpoch, // 10 iterations through whole dataset
            minCount, // 10 minimal count of words to include
            count2numSenses, //  thresholds for usings 2, 3,.. senses
            seed, // random seed
            local // true not use cluster
          )
          smod.setModelConst(
            negative, // 5 number of negative samples
            window, // 5 window size
            vectorSize, // 50 embedding size
            learningRate, // 0.025 beginning learning rate
            100, // multiplier for word number
            stepSize, // 10000 how many word words are processed before reducing learning rate
            oneSense, // indicator to use only 1 sense
            softMax, // indicator for softmax / sigmoid
            "",
            "",
            "",
            5, // interval for model saving
            1, // interval for model validation
            maxEmbNorm, // maximum square length of embedding. If larger the vectors are scaled down
            senseProbThresh, // re-initialize embedding if senseProbability is lower
            3, // level of training output
            0.1f, // weight reduction
            syn1OneSense) // syn1 has only one sense

          val rnd = new Random(seed)
          val vocabSiz = 5

          //------------- GENERATE the data -----------------------
          val wrds = Array("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
          require(vocabSiz <= wrds.length)
          val numSent = 9
          val numWordPerSent = 10
          val data = (0 until numSent)
            .map(x =>
              (0 until numWordPerSent).map(z =>
                wrds(rnd.nextInt(vocabSiz))) // generate the tokens
                .toArray.mkString(" ")
            ).toArray
          val textRDD: RDD[String] = sc.parallelize((data))
          println("generated a text with " + textRDD.count() + " sentences")

          //smod.
          if (softMax)
            smod.checkDerivSoftmax(textRDD)
          else
            smod.checkDerivSigmoid(textRDD)
        }
      }
    }
    sc.stop()
  }

  def doSigmoidTest(): Unit = {
    val conf = new SparkConf().setAppName("Sence2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
      conf.setMaster("local[2]")
    }
    val sc = new SparkContext(conf)
    val inputFile = ""
    val numRDD = 1
    val numEpoch = 10
    val minCount = 5
    val count2numSenses = "10" //  thresholds for usings 2, 3,.. senses
    val negative = 5
    val window = 5
    val vectorSize = 50
    val maxNumSenses = 2
    val minCountMultiSense = 10
    val seed = 42l
    val learningRate = 0.25f
    val stepSize = 10000
    val local = true
    val oneSense = false
    val softMax = true
    val maxEmbNorm = 5.0f
    val senseProbThresh = 0.01f
    val syn1OneSense = true // syn1 has only one sense


    val smod = new SenseAssignment(
      inputFile, // name of input file
      numRDD, // number of RDDs
      numEpoch, // 10 iterations through whole dataset
      minCount, // 10 minimal count of words to include
      count2numSenses, //  thresholds for usings 2, 3,.. senses
      seed, // random seed
      local // true not use cluster
    )
    smod.setModelConst(
      negative, // 5 number of negative samples
      window, // 5 window size
      vectorSize, // 50 embedding size
      learningRate, // 0.025 beginning learning rate
      100, // multiplier for word number
      stepSize, // 10000 how many word words are processed before reducing learning rate
      oneSense,
      softMax, // indicator for one Sense
      "",
      "",
      "",
      5, // interval for model saving
      2, // interval for model validation
      maxEmbNorm, // maximum square length of embedding. If larger the vectors are scaled down
      senseProbThresh, // re-initialize embedding if senseProbability is lower
      3, // level of training output
      0.1f, // weight decay
      syn1OneSense) // syn1 has only one sense

    val rnd = new Random(seed)
    val vocabSiz = 5

    //------------- GENERATE the data -----------------------
    val wrds = Array("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
    require(vocabSiz <= wrds.length)
    val numSent = 9
    val numWordPerSent = 10
    val data = (0 until numSent)
      .map(x =>
        (0 until numWordPerSent).map(z =>
          wrds(rnd.nextInt(vocabSiz))) // generate the tokens
          .toArray.mkString(" ")
      ).toArray
    val textRDD: RDD[String] = sc.parallelize((data))
    println("generated a text with " + textRDD.count() + " sentences")
    smod.learnVocab(textRDD) // generates vocabulary

    val (vocabSize, numberOfSensesPerWord) = smod.createNumberOfSensePerWord(rnd) // number of senses
    // for
    // each word

    val multiFreqDistr = smod.createMultinomFreqDistr()
    smod.mc.setDictInfo(vocabSize, numberOfSensesPerWord, multiFreqDistr)

    println("------- checking MultinomFreqDistr ---------------")
    val wordCounts = new Array[Int](vocabSize)
    for (i <- 0 until 1000000) {
      val iw = multiFreqDistr.next()
      wordCounts(iw) += 1
    }
    var sm0 = 0.0;
    var sm1 = 0.0
    val smoothedWordCounts = new Array[Double](vocabSize)
    for (iw <- 0 until vocabSize) {
      smoothedWordCounts(iw) = math.pow(smod.vocab(iw).cn * 1.0, smod.POWER)
      sm0 += smoothedWordCounts(iw)
      sm1 += wordCounts(iw)
    }
    var maxPdiff = 0.0
    for (iw <- 0 until vocabSize) {
      maxPdiff = math.max(maxPdiff, math.abs(smoothedWordCounts(iw) / sm0 - wordCounts(iw) / sm1))
    }
    println("MultinomFreqDistr maxPdiff=" + maxPdiff)
    require(maxPdiff < 1.0E-3, "NOT maxPdiff< 1.0E-3")


    val (syn0, syn1) = smod.initSynRand(rnd)
    val mu = new ModelUpdater(smod.mc, seed, syn0, syn1)

    println("checking stak, unstack")
    val syn0Stack = smod.mc.stackSynIntoArray(syn0)
    val syn00 = smod.mc.unstack(syn0Stack, 0, 1.0f)
    for (iw <- 0 until syn0.length) {
      for (is <- 0 until syn0(iw).length)
        for (j <- 0 until syn0(iw)(is).length)
          require(syn00(iw)(is)(j) == syn00(iw)(is)(j), iw + " " + is + " " + j)
    }

    var diffmax = 0.0
    for (x <- -smod.getMAX_EXP * 1.0f until smod.getMAX_EXP * 1.0f by 2.0f * smod.getMAX_EXP / smod.mc
      .getEXP_TABLE_SIZE) {
      val fappr = mu.sigmoidApprox(x)
      val f = mu.sigmoid(x)
      diffmax = math.max(diffmax, fappr - f)

    }
    println("sigmoid diffmax=" + diffmax + " between exact function and approximation")
    require(diffmax < 1.0E-7)
    sc.stop()

  }
}


/** Axel Poigne */
@SerialVersionUID(1000L)
object EmpiricalDerivative {

  val CON = 1.4
  val CON2 = CON * CON
  val NTAB = 10
  val SAFE = 2

  /** Returns the derivative of a function func at a point x by Ridders' method of polynomial
    * extrapolation.
    * Parameters: Stepsize is decreased by CON at each iteration. Max size of tableau is set by
    * NTAB. Return when error is SAFE worse than the best so far.
    *
    * @param f   a function of a vector DenseVector[Double] returning a double. May be enhanced by additional arguments
    * @param vec point vector   DenseVector[Double]
    * @param h   The value h is inp as an estimated initial stepsize; it need not be small,
    *            but rather should be an increment in x over which func changes substantially.
    * @return the estimated gradient and an estimate of the error in the derivative.
    * */
  implicit def apply(f: BDV[Double] => Double, vec: BDV[Double], h: Double): (BDV[Double], Double) = {
    require(h > 0, s"h == $h must be nonzero")
    val (grads, errs) = (0 until vec.length).map(apply(_, f, vec, h)).unzip
    (BDV(grads.toList: _*), Bmax(errs))
  }

  def apply(n: Int, f: BDV[Double] => Double, vec: BDV[Double], h: Double): (Double, Double) = {
    val a = BDM.zeros[Double](NTAB, NTAB)
    var err = Double.MaxValue
    var grad = 0.0
    var hh = h
    for (i <- 0 until NTAB if math.abs(a(i, i) - a(i - 1, i - 1)) < SAFE * err) {
      //# Successive columns in the Neville tableau will go to smaller
      hh /= CON //# stepsizes and higher orders of extrapolation.
      val xold = vec(n)
      vec(n) = xold + hh
      val fp = f(vec)
      vec(n) = xold - hh
      val fm = f(vec)
      vec(n) = xold
      a(0, i) = (fp - fm) / (2.0 * hh) //# Try new, smaller stepsize.
      var fac = CON2
      for (j <- 1 until i) {
        //# Compute extrapolations of various orders, requiring no new function evaluations.
        a(j, i) = (a(j - 1, i) * fac - a(j - 1, i - 1)) / (fac - 1.0)
        fac *= CON2
        val errt = Bmax(Babs(a(j, i) - a(j - 1, i)), Babs(a(j, i) - a(j - 1, i - 1)))
        if (errt <= err) {
          err = errt
          grad = a(j, i)
        }
      }
    }
    (grad, err)
  }

}

@SerialVersionUID(1000L)
object EmpDeriv {

  /** Returns the derivative of a function func at a point x by Ridders' method of polynomial
    * extrapolation.
    * Parameters: Stepsize is decreased by CON at each iteration. Max size of tableau is set by
    * NTAB. Return when error is SAFE worse than the best so far.
    *
    * @param func a function of a vector DenseVector[Double] returning a double. May be enhanced by additional arguments
    * @param h    The value h is inp as an estimated initial stepsize; it need not be small,
    *             but rather should be an increment in x over which func changes substantially.
    * @return the estimated gradient and an estimate of the error in the derivative.
    **/
  def adaptiveStepsizeVec(func: BDV[Double] => Double, vec: BDV[Double], h: Double, args: Any*): (
    BDV[Double], Double) = {
    val grad = BDV.zeros[Double](vec.length)
    val CON = 1.4
    val CON2 = CON * CON
    val BIG = 1.0E30
    val NTAB = 10
    val SAFE = 2
    if (h <= 0) throw new IllegalArgumentException("h=" + h + "  must be nonzero")
    var maxerr: Double = 0
    //    var fp = func(vec) // as a reference value
    var dfridr = 0.0
    for (idim <- 0 until vec.length) {
      val xold = vec(idim)
      val x = xold
      val a = BDM.zeros[Double](NTAB, NTAB) //np.zeros((NTAB, NTAB))
      val hh = h
      vec(idim) = x + hh
      val fp = func(vec)
      vec(idim) = x - hh
      val fm = func(vec)
      a(0, 0) = (fp - fm) / (2.0 * hh)
      var err = BIG
      var finished = false
      for (i <- 1 until NTAB if !finished) {
        //# Successive columns in the Neville tableau will go to smaller
        val hhc = hh / CON //# stepsizes and higher orders of extrapolation.
        vec(idim) = x + hhc
        val fp = func(vec)
        vec(idim) = x - hhc
        val fm = func(vec)
        a(0, i) = (fp - fm) / (2.0 * hhc) //# Try new, smaller stepsize.
        var fac = CON2
        for (j <- 1 until i) {
          //# Compute extrapolations of various orders, requiring no new function evaluations.
          a(j, i) = (a(j - 1, i) * fac - a(j - 1, i - 1)) / (fac - 1.0)
          fac = CON2 * fac
          val errt = math.max(math.abs(a(j, i) - a(j - 1, i)), math.abs(a(j, i) - a(j - 1, i - 1)))
          // The error strategy is to compare each new extrapolation to one order lower, both at
          // the present stepsize and the previous one.
          if (errt <= err) {
            //# If error is decreased, save the improved answer.
            err = errt
            dfridr = a(j, i)
          }
        }
        if (math.abs(a(i, i) - a(i - 1, i - 1)) >= SAFE * err) {
          grad(idim) = dfridr
          maxerr = math.max(maxerr, err)
          vec(idim) = xold
          finished = true
          // If higher order is worse by a significant factor SAFE, then quit early.
        }
      }
      grad(idim) = dfridr
      maxerr = math.max(maxerr, err)
      vec(idim) = xold
    }
    (grad, maxerr)
  }
}

object Ut {

  def BVtoString(v: BV[Double], fmt: String = "%10.3f"): String = {
    var s = ""
    for (i <- 0 until v.length) s += fmt.format(v(i)) + " "
    s
  }

  def printBV(name: String, v: BV[Double], maxlen: Int = -1, indent: String = "", fmt: String = "%10.3f"): String = {
    var s = name + "(0::" + v.length + ")= "
    if (maxlen > 0 && v.length > maxlen) {
      s += BVtoString(v(0 until maxlen), fmt)
    } else
      s += BVtoString(v, fmt)
    println(indent + s)
    s
  }


  @inline final def pp(vl: Double, fmt: String = "%10.3f"): String = {
    fmt.format(vl) + " "
  }

}