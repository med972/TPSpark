package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
     //Chemin relatif en partant du principe que tu auras les données au même emplacement (issue du preprocessor)
    val df_preprocessed = spark.read.parquet("data/prepared_trainingset")
    ///Users/mehdiregina/Documents/TP_ParisTech_2017_2018_starter/
    /** TF-IDF **/
      //séparer les textes en mots (ou tokens) avec un tokenizer
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //StopWordsRemover takes as input a sequence of strings (e.g. the output of a Tokenizer)
    // and drops all the stop words from the input sequences
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("words")


    //Prend en input des tokens mots (une liste de mots splités), retourne un vecteur
    // Vecteur contenant le nombre de mots différents dans la liste sous forme d'incdices et leur fréquence d'apparition dans 1 texte
    //les mots relevés semblent avoir le même indice sur l'ensemble du corpus de texte -> 2 infos
    //Info 1 : fequence d'appartion dans 1 texte donné, Info 2: frequence d'apparition dans le corpus de texte
    //le stage countVectorizer a un paramètre “minDF” qui permet de ne prendre que les mots apparaissant
    // dans au moins le nombre spécifié par minDF de textes
    val cvModel = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("rawFeatures")

    //tf-idf reflect the importance of a term to a document in the corpus
    //Term frequency TF(t,d)TF(t,d) is the number of times that term tt appears in document dd, (1 document = 1 text)
    // document frequency DF(t,D)DF(t,D) is the number of documents that contains term tt.
    //input count vectorizer renvoyant l'occurence d'un mot dans un document mais aussi dans un corpus de documents via 1 vecteur
    val idf = new IDF().setInputCol(cvModel.getOutputCol).setOutputCol("tfidf")


    /** CONVERT CATEGORIAL STRING FEATURES IN CATEGORIAL NUMERIC FEATURES **/
    //StringIndexer encodes a string column of labels to a column of label indices
    //Converti toute variable catégorielle en variable catégorielle numérique
    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /** VECTOR ASSEMBLER **/
    //SparkML Algo works with all features in one column
    //VectorAssembler is a transformer that combines a given list of columns into a single vector column.
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")


    /** MODEL **/
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    /** PIPELINE **/
      //Pipeline = Transfo données textuelles via tf-idf + formalisation (StringIndex,VecAssem) des données pour utiliser un algo
    val ModelPipeline = new Pipeline().setStages(Array(tokenizer,remover,cvModel,idf,countryIndexer,currencyIndexer,assembler,lr))
    val Model = ModelPipeline.fit(df_preprocessed)

    val lrDF = Model.transform(df_preprocessed)

    /** TRAINING AND GRID-SEARCH **/
    val Array(training, test) = df_preprocessed.randomSplit(Array(0.9, 0.1))

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.
    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF,Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(Math.pow(10,-8),Math.pow(10,-4),Math.pow(10,-2)))
      .build()

    val trainValidationSplit = new TrainValidationSplit() //PipelineModel + GridSearch
      .setEstimator(ModelPipeline)
      .setEvaluator(new MulticlassClassificationEvaluator() //Il faut préciser à l'évaluator ses paramètres !
        .setPredictionCol("predictions")
        .setLabelCol("final_status")
        .setMetricName("f1"))//support F1 default
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    //The model returned will be the model with the combination of params that give the best F1 score
    val gridModel = trainValidationSplit.fit(training)

    //Store the predictiction given by the best model and the previous data, in a new df
    val df_WithPredictions = gridModel.transform(test)
    df_WithPredictions.printSchema()

    df_WithPredictions.groupBy("final_status", "predictions").count.show()



  }
}
