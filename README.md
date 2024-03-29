# Resumen del proceso (Random Forest)

1. Se busca predecir la supervivencia en el accidente del Titanic utilizando Random Forest. Identificar qué tipo de personas era más probable que sobreviviera, utilizando datos de los pasajeros como el nombre, la edad, el género, la clase socioeconómica, etc.
2. El dataset contiene 11 variables y 891 observaciones, hay valores faltantes y clases a modificar. la variable target es Survived (1 si sobrevivió y 0 si no sobrevivió).
3. En el análisis exploratorio (aplicado solamente a la muestra de entrenamiento), se identifican variables a eliminar (Name, Cabin y Ticket) y se imputan valores faltantes en Age y Embarked. En el dataset de prueba (test) si bien no se modifican los datos, se eliminan las observaciones con datos faltantes en edad, ya que de otra manera se produce un error al querer predecir.
4. El sexo, la tarifa y la clase parecen ser variables importantes para predecir la supervivencia, lo cual es lógico ya que se sabe que priorizaron mujeres y las clases sociales altas. La correlación entre las variables no es demasiado alta, por ese lado no amerita eliminar variables.
5. En el paso 3 se estima un primer modelo de Random Forest con parámetros por defecto (en notebook explore_RandomForest.ipynb), luego se realiza una optimización aleatoria de parámetros para encontrar el modelo con el menor error. Se vuelve a estimar un modelo con los hiperparámetros óptimos y se lo guarda como modelo final. Tanto en el modelo inicial como en el optimizado se identifican como variables más importantes Age, Fare y Sex, lo cual confirma los hallazgos del EDA.

# Resumen del proceso (XGBoost)

1. En este ejercicio se utiliza el mismo dataset y repositorio que en el ejercicio sobre Random Forest. Primero se cargan los datasets ya preprocesados, y luego se aplica el modelo XGBoost.
2. Se estima un modelo XGBoost inicial y luego se optimiza sobre algunos parámetros. En ambos casos se grafica la importancia de las variables y el clasification report, se comparan los resultados.
3. El modelo con hiperparámetros optimizados no mejora la performance del modelo inicial, por lo que se elige al modelo inicial como el modelo final y se lo guarda.