# STEP 1: DATOS

# ejecutar en consola
# pip install pandas
# pip install matplotlib
# pip install seaborn
# pip install sklearn

# importar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import pickle

# cargar datos
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv', index_col=0)

# paso a categórica
df_raw.Survived = pd.Categorical(df_raw.Survived)
df_raw.Pclass = pd.Categorical(df_raw.Pclass)
df_raw.Name = pd.Categorical(df_raw.Name)
df_raw.Sex = pd.Categorical(df_raw.Sex)
df_raw.Cabin = pd.Categorical(df_raw.Cabin)
df_raw.Embarked = pd.Categorical(df_raw.Embarked)

# guardo data frame original
df_raw.to_csv('data/raw/datos_iniciales.csv', index = False)

# STEP 2: EDA

# se separan datos en X e y
y = df_raw['Survived']
X = df_raw.drop(columns = 'Survived')

# se separa en muestras de entrenamiento y evaluación
X_train , X_test , y_train , y_test = train_test_split(X, y, random_state = 3007)

# se crea data frame train para realizar EDA
df_train = pd.concat([X_train, y_train], axis = 1)

# se elimina variable que no aporta
df_train.drop(columns = 'Ticket', inplace = True)

# se elimina variable con muchos Null
df_train = df_train.drop(columns = 'Cabin')

# se imputa la media en faltantes de edad
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
# se imputa categoría más frecuente en Embarked
df_train['Embarked']=df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])

# se elimina variable name del dataset, ya que no aporta
df_train = df_train.drop(columns = 'Name')

# se recodifica (en train y test) para que tome valores numéricos (siguen siendo categóricas)
df_train['Sex'] = df_train['Sex'].cat.codes
df_train['Embarked'] = df_train['Embarked'].cat.codes

# guardo dataset procesado
df_train.to_csv('data/processed/datos_entrenamiento_procesados.csv', index = False)

# se vuelven a definir X_train e y_train para incorporar los ajustes
X_train = df_train.drop(columns = 'Survived')
y_train = df_train['Survived']

# elimino variables de X_test para que sea equivalente a X_train
X_test.drop(columns = ['Ticket', 'Name', 'Cabin'], inplace = True)

# recodifico categóricas
X_test['Sex'] = X_test['Sex'].cat.codes
X_test['Embarked'] = X_test['Embarked'].cat.codes

# elimino observaciones con NA en alguna variable (sino da error al predecir)
df_test = pd.concat([X_test, y_test], axis = 1)
df_test.dropna(inplace = True)
X_test = df_test.drop(columns = 'Survived')
y_test = df_test['Survived']

# creo dataset test actualizado y guardo
df_test = pd.concat([X_test, y_test], axis = 1)
df_test.to_csv('data/processed/datos_evaluacion_procesados.csv', index = False)

# STEP 3: APLICO RANDOM FOREST CON PARÁMETROS OPTIMIZADOS EN NOTEBOOK

mod_best = RandomForestClassifier(random_state = 3007, n_estimators = 300, min_samples_split = 5, min_samples_leaf = 2, max_depth = 10, criterion = 'entropy', bootstrap = False)
mod_best.fit(X_train, y_train)

# se guarda modelo final
filename = 'models/finalized_model.sav'
pickle.dump(mod_best, open(filename, 'wb'))