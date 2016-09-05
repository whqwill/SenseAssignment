import sbt.Keys._
import sbt._

object BuildSettings {

  lazy val buildSettings = Defaults.coreDefaultSettings ++ Seq(
    scalacOptions := Seq(
      "-deprecation",
      "-unchecked",
      "-language:_",
      "-target:jvm-1.8",
      "-encoding",
      "UTF-8",
      "-Xlint"
    ),

    javacOptions := Seq(
      "-Xlint:unchecked",
      "-Xlint:deprecation"
    )
  )
}

object Resolvers {
  // This is a temporary location within the Apache repo for the 1.0.0-RC3
  // release of Spark.
  val apache = "Apache Repository" at "https://repository.apache.org/content/repositories/orgapachespark-1012/"
  val typesafe = "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/"
  val sonatype = "Sonatype Release" at "https://oss.sonatype.org/content/repositories/releases"
  val mvnrepository = "MVN Repo" at "http://mvnrepository.com/artifact"
  val localmvn = "Local Maven Repository" at "file:///home/hwang/.m2/repository"

  val allResolvers = Seq(apache, typesafe, sonatype, mvnrepository, localmvn)
}



object Dependencies {
  object Version {
    //val Spark       = "1.5.2"
    val Spark = "1.6.0"
    //"1.4.1"
    //val Spark       = "1.3.0"
    //val ScalaTest   = "3.0.0-SNAP3"
    val ScalaTest     = "2.2.4"
    val ScalaCheck    = "1.11.3"
    val spray_json    = "1.2.6"
    val opennlp       = "1.5.3"
    val treetagger    = "1.1.1"
    val scalaIo       = "0.4.2"
  }

  val sparkCore      = "org.apache.spark"         %% "spark-core" % Version.Spark % "provided"
  val sparkSQL       = "org.apache.spark"         %% "spark-sql" % Version.Spark % "provided"
  val sparkStreaming = "org.apache.spark"         %% "spark-streaming" % Version.Spark % "provided"
  val sparkRepl      = "org.apache.spark"         %% "spark-repl" % Version.Spark % "provided"
  val sparkMllib     = "org.apache.spark"         %% "spark-mllib" % Version.Spark % "provided"
  val junitNovocode  = "com.novocode"             %% "junit-interface" % "0.11" % "test->default"

  val scalaIO        = "com.github.scala-incubator.io" %% "scala-io-file" % Version.scalaIo  withSources() // GP
  val scalaTest      = "org.scalatest"            %% "scalatest" % Version.ScalaTest % "test"
  val scalaCheck     = "org.scalacheck"           %% "scalacheck" % Version.ScalaCheck % "test"
  val netlib         = "com.github.fommil.netlib" % "all" % "1.1.2"
  val spray_json     = "io.spray"                 %% "spray-json"    % Version.spray_json withSources()  // AP
  val junit          = "junit"                    % "junit" % "4.11" % "test" //GP
  val opennlp        = "org.apache.opennlp"       % "opennlp-tools"    % Version.opennlp withSources()
  val treetagger     = "org.annolab.tt4j"         % "org.annolab.tt4j" % Version.treetagger withSources()
  val config         = "com.typesafe"             % "config" % "1.2.1"

  val allDependencies = Seq(sparkCore, sparkSQL, sparkStreaming, sparkMllib, scalaTest, scalaCheck, netlib,
    spray_json, junit, opennlp, treetagger, config)
}

object SparkBuild extends Build {

  lazy val spark_examples = Project(
    id = "SparkWord2Vec",
    base = file("."),
    settings = BuildSettings.buildSettings ++ Seq(
      // runScriptSetting,
      resolvers := Resolvers.allResolvers,
      libraryDependencies ++= Dependencies.allDependencies,
      unmanagedResourceDirectories in Compile += baseDirectory.value / "conf",
      unmanagedResourceDirectories in Compile += baseDirectory.value / "lib",
      mainClass := Some("de.fraunhofer.iais.kd.haiqing.Main_sense"),
      // Must run Spark tests sequentially because they compete for port 4040!
      parallelExecution in Test := false
    )
  )
}
