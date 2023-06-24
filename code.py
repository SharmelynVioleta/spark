import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

#Creamos una sesion de spark
spark = SparkSession.builder.appName("CSV Peliculas").getOrCreate()

#Cargamos el dataset
pandasDF = pd.read_csv('dataset.zip')

df = spark.createDataFrame(pandasDF)

# Seleccionamos los campos
df = df.select("userId", "movieId", "rating").withColumnRenamed("movieId","pelicula").withColumnRenamed("userId","usuario")
df = df.withColumn("rating", df["rating"].cast("float"))
df = df.filter(df["usuario"].isNotNull() & df["pelicula"].isNotNull() & df["rating"].isNotNull())

# dividimos la data en training y test
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

user_indexer = StringIndexer(inputCol="usuario", outputCol="userIndex")
movie_indexer = StringIndexer(inputCol="pelicula", outputCol="movieIndex")
indexed_data = user_indexer.fit(train_data).transform(train_data)
indexed_data_final = movie_indexer.fit(indexed_data).transform(indexed_data)

# Creamos el modelo de recomendacion ASL 
als = ALS(userCol="userIndex", itemCol="movieIndex", ratingCol="rating", nonnegative=True)

# Ajustamos el modelo con los datos de entrenamiento
model = als.fit(indexed_data_final)

user_indexer = StringIndexer(inputCol="usuario", outputCol="userIndex")
movie_indexer = StringIndexer(inputCol="pelicula", outputCol="movieIndex")
indexed_test_data = user_indexer.fit(test_data).transform(test_data)
indexed_test_data = movie_indexer.fit(indexed_test_data).transform(indexed_test_data)

#Testeamos con RMSE
predictions = model.transform(indexed_test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

# Generar 4 recomendaciones de peliculas por usuario
recommendations = model.recommendForUserSubset(indexed_test_data, 4)
print(recommendations.show(15))
print("rmse: " + str(rmse))