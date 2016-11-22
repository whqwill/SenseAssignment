# SenseAssignment

How to use jar


The path of jar file :
/SOME_PATH/SenseAssignment.jar


An example of submitting a job to train sense embedding:
/opt/spark/bin/spark-submit --master local[*] --conf "spark.driver.maxResultSize=110g" --driver-memory 110g  /SOME_PATH/SenseAssignment.jar file:///SOME_PATH/DATA/DATA.txt 20 5_10 200 10 5 200 2000_10000 0.1 0.9 true /SOME_PATH/DATA/evaluationWords /SOME_PATH/RESULT/r1sense /SOME_PATH/RESULT/rmsense​


Explanation: 
Submitting File:
/opt/spark/bin/spark-submit : the submitting bin file of spark , which can be at other place like /home/IAIS/hwang/spark/bin/spark-submit (my home folder). You can copy the folder “/opt/spark” to your own home folder and use it.

Configuration of Spark:
--master local[*] : local model with all cpu cores (one machine)
--conf "spark.driver.maxResultSize=110g" : Limit of total size of serialized results of all partitions for each Spark action (e.g. collect)
--driver-memory 110g : Amount of memory to use for the driver process. Notice that for the local model, you don't have to specify the amount of memory to use per executor process (--executor.memory), because everything is running in one Java Virtual Machine and “--driver-memory” is used to specify the size limit of it. 

Jar file:
/SOME_PATH/SenseAssignment.jar : the jar file

Arguments of jar file:
args(0) = file:///SOME_PATH/DATA/DATA.txt : the path of training file. Notice that it's better to use "file:///". 
args(1) = 20: the number of RDDs. For each iteration, the program uses only one RDD to train embedding.
args(2) = 5_10: 5 is the maximum epochs for word embedding, 10 is the maximum epochs for sense embedding. 
args(3) = 200: the minimum count for words in dictionary.
args(4) = 10: the number of negative samples 
args(5) = 5: the window size
args(6) = 200: the vector size of embedding. 
args(7) = 2000_10000: 2000 is the minimum count for words with 2 senses,  10000 is the minimum count for words with 3 senses.
args(8) = 0.1: the beginning learning rate
args(9) = 0.9: the reduction factor of learning rate. After each iteration, the learning rate will be reduced by this factor.
args(10) = true: local model (one machine). “false” means cluster model (several machines).
args(11) = /SOME_PATH/DATA/evaluationWords: the path of evaluation words from SCWS and word353, which will be added into the dictionary.
args(12) = /SOME_PATH/RESULT/r1sense: the output path for the result of normal word embedding (each word have only one sense). And the result will be used as the initialization of sense embedding later. 
args(13) = /SOME_PATH/RESULT/rmsense: the output path for the result of sense embedding (each word can have several senses)


Tips for log files:

You can use command “nohup” + “&” to let the program running in the background:

nohup /opt/spark/bin/spark-submit --master local[*] --conf "spark.driver.maxResultSize=110g" --driver-memory 110g  /SOME_PATH/SenseAssignment.jar file:///SOME_PATH/DATA/DATA.txt 20 5_10 200 10 5 200 2000_10000 0.1 0.9 true /SOME_PATH/DATA/evaluationWords /SOME_PATH/RESULT/r1sense /SOME_PATH/RESULT/rmsense > /SOME_PATH/LOG/logFile 2> /SOME_PATH/LOG/logFile_err &

> /SOME_PATH/LOG/logFile : print the standard output to “logFile”. 
2> /SOME_PATH/LOG/logFile_err : print the error output to “logFile_err”. 


Test Nearest Words:
/opt/spark/bin/spark-submit --master local[*] --conf "spark.driver.maxResultSize=110g" --driver-memory 110g --class de.fraunhofer.iais.kd.haiqing.TestSenseVectors /SOME_PATH/SenseAssignment.jar /SOME_PATH/RESULT/rmsense 10 apple bank day net

--class: the main class to implement. The default is “de.fraunhofer.iais.kd.haiqing.Main_sense” (train sense embedding)

Arguments of jar file:
args(0) = /SOME_PATH/RESULT/rmsense : the result path of sense embedding
args(1) = 10 : the number of nearest words 
args(2) ... args(n) = apple bank day net : words for calculating nearest words




