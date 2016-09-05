package de.fraunhofer.iais.kd.haiqing

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by hwang on 09.02.16.
  * recommended VM-Options : -Xmx13g -Dspark.executor.memory=2g
  * -Dspark.driver.memory=2g  should be used if operating locally
  * -Dspark.cores.max=xxx 	(not set) 	When running on a standalone deploy cluster or a Mesos cluster in
  * "coarse-grained" sharing mode, the maximum amount of CPU cores to request for the application from across
  * the cluster (not from each machine). If not set, the default will be spark.deploy.defaultCores on Spark's
  * standalone cluster manager, or infinite (all available cores) on Mesos.
  */
object Main_sense {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN) // omit all iteration logging
    Logger.getLogger("akka").setLevel(Level.WARN) // omit all iteration logging
    val conf = new SparkConf().setAppName("Sense2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
      conf.setMaster("local[4]")
    }
    val sc = new SparkContext(conf)
    println("sc.defaultParallelism=" + sc.defaultParallelism + "   " + sc.master)
    val argNam = Array("inpFile", "numRdd", "numEpoch", "minCount", "numNegative", "windowSize", "vectorSize",
      "count2numSenses", "learnRate", "stepSize", "local", "evaluationWordPath", "1-senseModelPath", "multi-senseModelPath")
    for (i <- 0 until args.length)
      println("arg(" + i + ") " + argNam(i) + " = " + args(i))

    // /home/gpaass/data/corpora/enNews2013/news.2013.en.shuffled.001fraction 10 50 10 20 5 50 20_60 0.02 10000 true
    // /home/gpaass/Dokumente/research/masterWang/resultsData/r1sense /home/gpaass/Dokumente/research/masterWang/resultsData/rmsense

    // /home/gpaass/data/corpora/enNews2013/news.2013.en.shuffled.01fraction 100 1_5 7 15 5 80 50_250 0.02 10000 true
    // /home/gpaass/Dokumente/research/masterWang/resultsData/r1sense /home/gpaass/Dokumente/research/masterWang/resultsData/rmsense

     // input file should contain lines of text, each line containing a sentence
    val input = sc.textFile(args(0), sc.defaultParallelism)
    println("input.getNumPartitions="+input.getNumPartitions)
    println("sc.defaultParallelism="+sc.defaultParallelism)
    val lines = input.take(2)
    println("---- read " + input.count() + " lines of text from file " + args(0)+ " ----")
    for (i <- 0 until lines.length)
      println("   "+lines(i))
    val ENCODE = 100
    //val oneSense = (args.length == 12)
    val softMax = false // softMax or sigmoid
    val modelPathMultiSense = if (args.length < 14) "" else args(13)
    val modelSaveIter = 5 // save the model after this number of iterations
    val validationRation = 0.1f // max. fraction of data to use for validation
    val modelValidateIter = 5 //  validate the model after this number of iterations
    val validationRatio = 0.1f // maximum fraction of data to use for validation
    val maxValidationSize: Int = 200000 // maximum number of sentences for validation
    val validationIsSubset = false // select validationset as subset of trainingssets
    val maxEmbNorm = 15.0f
    val senseProbThresh = 0.02f // re-initialize embedding if senseProbability is lower
    val printLv=3 // level for training output
    val weightDecay = 0.1f
    val syn1OneSense:Boolean = true // syn1 has only one sense

    // 2_5 -> number iterations one sense, number of iterations multisense
    // 0_R5 -> read current syn0 syn1 of multisense and do another 5 iterations
    val parts=args(2).split("_")
    require(parts.length==2,"args(2)="+args(2)+"must have two parts separated by _")
    val (readOneSense,numEpochOneSense) = if(parts(0).charAt(0)=='R'){
      (true,parts(0).substring(1).toInt)
    } else
      (false,parts(0).toInt)


    val (readMultiSense,numEpochMultiSense) = if(parts(1).charAt(0)=='R'){
      (true,parts(1).substring(1).toInt)
    } else
      (false,parts(1).toInt)
    if(readOneSense)
      require(numEpochOneSense>0 && !readMultiSense,"NOT numEpochOneSense>0 && !readMultiSense")
    if(readMultiSense)
      require(numEpochOneSense==0 && numEpochMultiSense>0,"NOT numEpochOneSense==0 && numEpochMultiSense>0")
    val st1 = if(readOneSense) "read previous params" else "random init"
    println("one sense:   numEpoch = "+numEpochOneSense+  ". init parameters: "+st1)
    val st2 = if(readMultiSense) "read previous multisense params" else "read previous onesense params"
    println("multi sense: numEpoch = "+numEpochMultiSense+". init parameters: "+st2)

   if(numEpochOneSense>0) {
      val oneSense = true
      val senseModel = new SenseAssignment(
        args(0), // inputfile
        args(1).toInt, // numRdds
        numEpochOneSense, // numEpoch = iterations through whole dataset
        args(3).toInt, // minCount = minimal count of words to include
        args(7), // thresholds for count -> number of senses val count2numSenses:Array[Int],
        42l, // seed
        args(10).toBoolean, // true not use cluster
        validationRatio,
        maxValidationSize, // maximum number of sentences for validation
        validationIsSubset, // true if validation set is subset of trainingset
        readOneSense    // read params from file
      )
      senseModel.setModelConst(
        args(4).toInt, // 5 number of negative samples
        args(5).toInt, // 5 window size
        args(6).toInt, // 50 embedding size
        args(8).toFloat, // 0.025 beginning learning rate
        ENCODE, // multiplier for word number, mu
        args(9).toFloat, // 10000 how many word words are processed before reducing learning rate
        oneSense, // indicator for using only 1 sense
        softMax, // indicator for sftMax or sigmoid activation
        args(11), // the file of evaluation Words
        args(12), //synPath path with stored model with 1 sense
        modelPathMultiSense, //outputPath path to write multisense model
        modelSaveIter, // save the model after this number of iterations
        modelValidateIter, // validate the model after this number of iterations
        maxEmbNorm, // maximum square length of embedding. If larger the vectors are scaled down
        senseProbThresh, //re-initialize embedding if senseProbability is lower
        printLv, // level for training output
        weightDecay, // weight reduction
        syn1OneSense) // syn1 has only one sense

      senseModel.trainWrapper(
        input //outputPath path to write multisense model
      )
    }

    if(numEpochMultiSense>0) {
      val oneSense = false
      val senseModel = new SenseAssignment(
        args(0), // inputfile
        args(1).toInt, // numRdds
        numEpochMultiSense, // numEpoch = iterations through whole dataset
        args(3).toInt, // minCount = minimal count of words to include
        args(7), // thresholds for count -> number of senses val count2numSenses:Array[Int],
        42l, // seed
        args(10).toBoolean, // true not use cluster
        validationRatio,
        maxValidationSize, // maximum number of sentences for validation
        validationIsSubset, // true if validation set is subset of trainingset
        readMultiSense    // read params from file
      )
      senseModel.setModelConst(
        args(4).toInt, // 5 number of negative samples
        args(5).toInt, // 5 window size
        args(6).toInt, // 50 embedding size
        args(8).toFloat, // 0.025 beginning learning rate
        ENCODE, // multiplier for word number, mu
        args(9).toFloat, // 10000 how many word words are processed before reducing learning rate
        oneSense, // indicator for using only 1 sense
        softMax, // indicator for sftMax or sigmoid activation
        args(11), // the file of evaluation Words
        args(12), //synPath path with stored model with 1 sense
        modelPathMultiSense, //outputPath path to write multisense model
        modelSaveIter, // save the model after this number of iterations
        modelValidateIter, // validate the model after this number of iterations
        maxEmbNorm, // maximum square length of embedding. If larger the vectors are scaled down
        senseProbThresh, //re-initialize embedding if senseProbability is lower
        printLv, // level for training output
        weightDecay, // weight reduction
        syn1OneSense) // syn1 has only one sense
      senseModel.trainWrapper(
        input //outputPath path to write multisense model
      )
    }
  }
}

// epoch= 1 iRDD=0/10 VALISET: lossPerPredict	=  0,179035 with learnRate = 0.01
// epoch= 1 iRDD=0/10 VALISET: lossPerPredict	=  0,176426 with learnRate = 0.02
//arg(1) numRdd = 10
//arg(2) numEpoch = 3
//arg(3) minCount = 10
//arg(4) numNegative = 20
//arg(5) windowSize = 5
//arg(6) vectorSize = 50
//arg(7) count2numSenses = 20_60
//arg(8) learnRate = 0.01
//arg(9) gamma = 0.95
//arg(10) local = true
//
