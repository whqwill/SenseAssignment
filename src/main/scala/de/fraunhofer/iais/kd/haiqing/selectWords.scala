package de.fraunhofer.iais.kd.haiqing

/**
 * Created by hwang on 17.06.16.
 */
object selectWords {
  def main(args: Array[String]): Unit = {
    require(args.length>=3,"args.length<3")
    val synFileName = args(0)+"/syn0.txt"
    val numSynonyms = args(1).toInt
    println("----- determine "+numSynonyms+" closest embeddings from file "+synFileName)
    val (word2numSense, wordSense2ind, senseVec, vectorSize) = SenseAssignment.readSynToVector(synFileName)
    val model = new Word2VecModel(wordSense2ind, senseVec)

    val searchTerms = (2 until args.length).map(i => args(i)).toArray

    val cosineDist = true
    model.saveNeighbors(searchTerms, word2numSense, numSynonyms,cosineDist,args(0))

  }

}
