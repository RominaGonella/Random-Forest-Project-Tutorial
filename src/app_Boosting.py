# Este proyecto retoma el dataset utilizado en el proyecto anterior (Random Forest), por lo que se utilizan los datos preprocesados.

# importar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pickle

# se cargan archivos preprocesados
df_train = pd.read_csv('data/processed/datos_entrenamiento_procesados.csv')
df_test = pd.read_csv('data/processed/datos_evaluacion_procesados.csv')

# creo X e y (train y test)
X_train = df_train.drop(columns = 'Survived')
y_train = df_train['Survived']

X_test = df_test.drop(columns = 'Survived')
y_test = df_test['Survived']

## STEP 1: CONSTRUIR MODELO XGBOOST

# creo objetos
D_train = xgb.DMatrix(X_train, label = y_train)
D_test = xgb.DMatrix(X_test, label = y_test)

# defino parámetros
param = {
    'gamma': 0,
    'eta': 0.3, # learning rate
    'max_depth': 5,
    'objective': 'multi:softmax',  # para clasificación
    'num_class': 2, # cantidad de clases de target
    'seed': 3107}  # semilla

# cantidad de iteraciones
steps = 20

# modelo
mod_xgb = xgb.train(param, D_train, steps)

## STEP 2: EVALUACIÓN Y OPTIMIZACIÓN DE PARÁMETROS

# predicción con muestra test
y_pred = mod_xgb.predict(D_test)

# se aplica grid search para optimizar hiperparámetros
clf = xgb.XGBClassifier(steps = 20)
parameters = {
     "eta" : [0.10, 0.30] , # learning rate
     "max_depth" : [5, 10, 15],
     "gamma" : [ 0.0, 0.4, 0.6, 0.8],
     'objective': ['multi:softmax'],  # para clasificación
     'num_class': [2], # cantidad de clases de target
     "seed" : [3107], # semilla
     "loss": ['log_loss', 'deviance', 'exponential'],
     "criterion": ['friedman_mse', 'squared_error', 'mse'],
     }
grid = GridSearchCV(clf, parameters, cv = 5, n_jobs = -1) # n_jobs = -1 implica usar todos los procesadores
grid.fit(X_train, y_train)

best_param_xgb = grid.best_params_
best_param_xgb

# usando la mejor combinación de hiperparámetros, estimo modelo final
mod_xgb_best = xgb.XGBClassifier(**best_param_xgb, steps = 20)
mod_xgb_best.fit(X_train, y_train)

# predicción con muestra test
y_pred = mod_xgb_best.predict(X_test)

# LA OPTIMIZACIÓN NO MEJORA LA PERFORMANCE DEL MODELO INICIAL, ME QUEDO CON MODELO INICIAL

# se guarda modelo final
filename = 'models/xgb_finalized_model.sav'
pickle.dump(mod_xgb, open(filename, 'wb'))