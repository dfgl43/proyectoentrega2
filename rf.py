"""
Created on 2024-11-10

@author: Equipo 18
"""

# Se importan las librerías y los datos del proyecto 
import pandas as pd
import numpy as np
df = pd.read_csv('Steam_2024_bestRevenue_1500.csv')

# Se eliminan las columnas categóricas no deseadas excepto 'publisherClass'
df = df.drop(['name', 'releaseDate', 'publishers', 'developers', 'steamId'], axis=1)

# Aplicar transformación logarítmica a características con alta variabilidad
df['log_copiesSold'] = np.log1p(df['copiesSold'])

# Realizar One-Hot Encoding a la columna 'publisherClass'
df_encoded = pd.get_dummies(df, columns=['publisherClass'], drop_first=True)

# Se definen las características (X), la variable objetivo continua (y) y se parten los datos en train y test
X = df_encoded.drop('revenue', axis=1) 
y = df_encoded['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Se importa MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Se define el servidor para llevar el registro de modelos y artefactos
mlflow.set_tracking_uri('http://localhost:5000')
# Se registra el experimento
experiment = mlflow.set_experiment("proyecto-dsa")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Se definen los parámetros iniciales del modelo encontrados en la primera exploración local
    n_estimators = 300
    max_depth = 20
    min_samples_leaf = 3
    min_samples_split = 2
    # Se crea el modelo con los parámetros definidos y se entrena
    rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split)
    rf.fit(X_train, y_train)
    # Realice predicciones de prueba
    predictions = rf.predict(X_test)
  
    # Registre los parámetros
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("minsamplesleaf", min_samples_leaf)
    mlflow.log_param("minsamplessplit", min_samples_split)
  
    # Registre el modelo
    mlflow.sklearn.log_model(rf, "random-forest-model")
  
    # Cree y registre la métrica de interés
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    print(mse)
