import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession

// create a Spark session
val spark = SparkSession.builder().appName("RandomForestClassifierExample").getOrCreate()

// load the dataset and create a dataframe
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// converting the target column into categorical variable
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// identify categorical features and transform them to numerical features
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// split the data into training and testing set
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// train the classifier using RandomForest
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// convert the indexed labels back to original labels
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// pipeline the stages of indexing label, feature indexing and Random Forest classifier
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// fit the pipeline model to the training data
val model = pipeline.fit(trainingData)

// make predictions on the test data
val predictions = model.transform(testData)

// evaluate the model using MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)

// print the accuracy of the model
println("Accuracy: " + accuracy)

// stop the Spark session
spark.stop()
