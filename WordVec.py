from pyspark import SparkConf
import pyspark.sql.functions as fc
from pyspark.sql.functions import round, col
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


sc = SparkContext('local')
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("WARN")


trending = spark.read.csv(path ="trending_sub.csv",header=True)
#Having a general look at the dataframe
trending.printSchema()

##
  ## Extracting EMbedding from sentence training
  ##
object TextEmbedding {
  val embeddingSize = 3

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TextEmbedding")
      .master("local[2]")
      .getOrCreate()

    val df = loadText(spark)
    val model = trianByWord2Vec(spark, df)
    saveModel(spark, model, args(0))
    saveTextEmb(spark, model, df, args(1))
    saveWordEmb(spark, model, args(2))

    loadModel(spark, args(0))

  }

  ## Setting up socket
  def loadText(spark: SparkSession): DataFrame = {
    val df = trending.text.toDF("id", "words")
    println("df count:" + df.count())
    df.show(10, false)
    df
  }

  ## Process of Training
  def trianByWord2Vec(spark: SparkSession, df: DataFrame): Word2VecModel = {
    val wordDataFrame = df
    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("result")
      .setVectorSize(embeddingSize)
      .setMinCount(3)
      .setWindowSize(3)
      .setMaxIter(10)

    val model = word2Vec.fit(wordDataFrame)

    model.getVectors.show(100, false)
    //    +----------+------------------------------------------------------------------+
    //    |word      |vector                                                            |
    //    +----------+------------------------------------------------------------------+
    //    |holiday   |[-0.053989291191101074,0.14687322080135345,-0.0022512583527714014]|
    //    +----------+------------------------------------------------------------------+

  }

  ## Saving w2v model
  def saveModel(spark: SparkSession, model: Word2VecModel, path:String): Unit = {
    println("saving model:" + path)
    model.write.mode(SaveMode.Overwrite).save(path)
  }

  ## Saving word embedding
  def saveTextEmb(spark: SparkSession, model: Word2VecModel, df: DataFrame, path:String): Unit = {
    println(s"saving $path")
    val result = model.transform(df)
    result.printSchema()
    result.show(false)

    result.select("id", "result")
      .repartition(5)
      .write
      .option("sep", "\t")
      .mode(SaveMode.Overwrite)
      .parquet(path)

  }

  ## loading Word2Vec Model and configuring
  def loadModel(spark:SparkSession, path:String):Word2VecModel = {
    val model = Word2VecModel.load(path)

    model.getVectors.show(false)
    //    +----------+------------------------------------------------------------------+
    //    |word      |vector                                                            |
    //    +----------+------------------------------------------------------------------+
    //    |happy     |[-0.053989291191101074,0.14687322080135345,-0.0022512583527714014]|
    //    |tutorial  |[-0.16293057799339294,-0.14514029026031494,0.1139335036277771]    |
    //    |funny     |[-0.0406828410923481,0.028049567714333534,-0.16289857029914856]   |
    //    |dance     |[-0.1490514725446701,-0.04974571615457535,0.03320947289466858]    |
    //    |I         |[-0.019095497205853462,-0.131216898560524,0.14303986728191376]    |
    //    |selfish   |[0.16541987657546997,0.06469681113958359,0.09233078360557556]     |
    //    |lucky     |[0.036407098174095154,0.05800342187285423,-0.021965932101011276]  |
    //    |boyfriend |[-0.1267719864845276,0.09859133511781693,-0.10378564894199371]    |
    //    |could     |[0.15352481603622437,0.06008218228816986,0.07726015895605087]     |
    //    |use       |[0.08318991959095001,0.002120430115610361,-0.07926633954048157]   |
    //    |Hi        |[-0.05663909390568733,0.009638422168791294,-0.033786069601774216] |
    //    |girou     |[0.11912573128938675,0.1333899050951004,0.1441687047481537]       |
    //    |case      |[0.14080166816711426,0.08094961196184158,0.1596144139766693]      |
    //    |about     |[0.11579915136098862,0.10381520539522171,-0.06980287283658981]    |
    //    |minute    |[0.12235434353351593,-0.03189820423722267,-0.1423865109682083]    |
    //    |wish      |[0.14934538304805756,-0.11263544857501984,-0.03990427032113075]   |
    //    +----------+------------------------------------------------------------------+

    println("model.getVectors.printSchema()")
    model.getVectors.printSchema()

    val embMap = model.getVectors
      .collect()
      .map(row => {
        val value = row.getAs[Vector](1)
        val key = row.getAs[String](0)
        (key, value)
      }).toMap

    ## Similarity test
    model.findSynonyms("I", 2).show(false)
    //    +-------+-------------------+
    //    |word   |similarity         |
    //    +-------+-------------------+
    //    |am     |0.800910234451294  |
    //    |but    |0.45088085532188416|
    //    +-------+-------------------+

    model

  }

}

