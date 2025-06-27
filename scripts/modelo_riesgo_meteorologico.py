# %% [markdown]
# # Modelo de Clasificación de Días de Riesgo Meteorológico

# %% [markdown]
# ## 1. Importación de librerias y carga de datos

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el archivo de datos desde data/raw
df = pd.read_csv('../data/raw/data_cañete.txt', sep='\\s+', header=None)

# Asignar nombres de columnas
df.columns = ['anio', 'mes', 'dia', 'precipitacion', 'temp_max', 'temp_min']

# %% [markdown]
# ## 2. Análisis Exploratorio de Datos (EDA)

# %% [markdown]
# ### Ver estructura y tipos de datos

# %%
# Mostrar información básica del dataset
print("Información del dataset:")
print(f"Forma del dataset: {df.shape}")
print(f"Columnas: {list(df.columns)}")
print("\nPrimeras 5 filas:")
print(df.head())
print("\nTipos de datos por columna")
print(df.dtypes)

# %% [markdown]
# ### Verificar datos nulos

# %%
df.isnull().sum()

# %% [markdown]
# ### Estadísticas descriptivas

# %%
df.describe()

# %% [markdown]
# ### Visualización de distribuciones

# %% [markdown]
# #### Histograma

# %%
axes = df[['precipitacion', 'temp_max', 'temp_min']].hist(bins=20, figsize=(12,6))

# Personalizar cada gráfico
axes[0, 0].set_xlabel("Precipitación (mm)")
axes[0, 0].set_ylabel("Número de días")
axes[0, 0].set_title("Distribución de Precipitación")

axes[0, 1].set_xlabel("Temperatura Máxima (°C)")
axes[0, 1].set_ylabel("Número de días")
axes[0, 1].set_title("Distribución de Temp. Máxima")

axes[1, 0].set_xlabel("Temperatura Mínima (°C)")
axes[1, 0].set_ylabel("Número de días")
axes[1, 0].set_title("Distribución de Temp. Mínima")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Boxplots

# %%
plt.figure(figsize=(10,5))
sns.boxplot(data=df[['precipitacion', 'temp_max', 'temp_min']])
plt.title('Boxplot de variables meteorológicas')
plt.show()

# %% [markdown]
# ### Matriz de Correlación

# %%
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlación entre variables')
plt.show()

# %% [markdown]
# ### Variable de Clasificación

# %%
# Crear una nueva columna 'riesgo_meteorologico' basada en las condiciones especificadas
df['riesgo_meteorologico'] = np.where(
    (df['temp_max'] >= 33) | (df['precipitacion'] >= 15),
    1,  # Día de riesgo
    0   # Día normal
)

# Ver la cantidad de valores de riesgo meteorológico
print("\nConteo de días con riesgo meteorológico:")
print(df['riesgo_meteorologico'].value_counts())

# %% [markdown]
# ## 3. Preprocesamiento de datos

# %% [markdown]
# ### Reemplazar valores -99.9 por NaN

# %%
# Reemplazar valores erróneos por NaN
df[['precipitacion', 'temp_max', 'temp_min']] = df[['precipitacion', 'temp_max', 'temp_min']].replace(-99.9, np.nan)

# %%
# Contar valores faltantes por columna
df.isnull().sum()

# %%
# Imputar valores faltantes con la media de cada columna
df['temp_max'] = df['temp_max'].fillna(df['temp_max'].mean())
df['temp_min'] = df['temp_min'].fillna(df['temp_min'].mean())
df['precipitacion'] = df['precipitacion'].fillna(df['precipitacion'].mean())

# %%
# Volver a validar valores nulos
df.isnull().sum()

# Volver a mostrar la información del dataset después de las imputaciones
df.describe()

# %% [markdown]
# ### Volver a generar la variable de clasificación

# %%
# Eliminar la columna si ya existe
if 'riesgo_meteorologico' in df.columns:
    df.drop(columns='riesgo_meteorologico', inplace=True)

# Crear de nuevo la columna ya con datos limpios
df['riesgo_meteorologico'] = np.where(
    (df['temp_max'] >= 33) | (df['precipitacion'] >= 15),
    1,
    0
)

# Ver la cantidad de valores de riesgo meteorológico
print("\nConteo de días con riesgo meteorológico:")
print(df['riesgo_meteorologico'].value_counts())

# %% [markdown]
# ### Guardar los datos limpios

# %%
# Guardar los datos limpios en data/clean
df.to_csv('../data/clean/data_cañete_clean.csv', index=False)

# %% [markdown]
# ## 4. Train/Test Split

# %% [markdown]
# ### Separar variables predictoras y variable objetivo

# %%
# X = variables independientes (features)
X = df[['precipitacion', 'temp_max', 'temp_min']]

# y = variable objetivo
y = df['riesgo_meteorologico']

# %% [markdown]
# ### Dividir en conjunto de entrenamiento y prueba

# %%
from sklearn.model_selection import train_test_split

# Dividir (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)


# %% [markdown]
# ### Verificar dimensiones

# %%
print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)
print("Distribución en entrenamiento:", y_train.value_counts())
print("Distribución en prueba:", y_test.value_counts())


# %% [markdown]
# ## 5. Entrenamiento del modelo

# %% [markdown]
# ### Importar y entrenar el modelo

# %%
from sklearn.tree import DecisionTreeClassifier

# Crear el modelo
modelo_arbol = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)

# Entrenarlo con el conjunto de entrenamiento
modelo_arbol.fit(X_train, y_train)

# %% [markdown]
# ### Hacer predicciones

# %%
# Usamos el conjunto de prueba (X_test)
y_pred = modelo_arbol.predict(X_test)

# %% [markdown]
# ### Evaluar el modelo

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Precisión general
print("Accuracy:", accuracy_score(y_test, y_pred))

# Reporte detallado
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Matriz de confusión
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))

# %% [markdown]
# ### Matriz de Confusión

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calcular matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Mostrarla como imagen
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión")
plt.grid(False)
plt.show()


# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %%
# Visualizar el árbol de decisión
from sklearn.tree import plot_tree
plt.figure(figsize=(12,8))
plot_tree(modelo_arbol, 
          feature_names=X.columns,
          class_names=['Normal', 'Riesgo'],
          filled=True, 
          rounded=True,
          fontsize=12)
plt.title('Árbol de Decisión para Riesgo Meteorológico')
plt.show()

# %% [markdown]
# ## Comparación con otros modelos

# %% [markdown]
# ### Random Forest

# %% [markdown]
# #### Importar y crear el modelo

# %%
from sklearn.ensemble import RandomForestClassifier

modelo_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)

# %% [markdown]
# #### Entrenar el modelo

# %%
modelo_rf.fit(X_train, y_train)

# %% [markdown]
# #### Predecir con el conjunto de prueba

# %%
y_pred_rf = modelo_rf.predict(X_test)

# %% [markdown]
# #### Evaluar el modelo

# %%
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Reporte de métricas
print(classification_report(y_test, y_pred_rf))

# Matriz de confusión
disp = ConfusionMatrixDisplay.from_estimator(
    modelo_rf, X_test, y_test, display_labels=['Normal', 'Riesgo'],
    cmap='Blues'
)
disp.ax_.set_title("Matriz de Confusión - Random Forest")
plt.show()

# %% [markdown]
# ## Conclusiones

# %% [markdown]
# ### Importancia de variables

# %%
# Obtener la importancia
importancias = modelo_rf.feature_importances_

# Asociarlas con sus nombres
features = X.columns
df_importancia = pd.DataFrame({'Variable': features, 'Importancia': importancias})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=True)

# %% [markdown]
# ### Graficar la importancia

# %%
plt.figure(figsize=(8, 4))
plt.barh(df_importancia['Variable'], df_importancia['Importancia'], color='teal')
plt.title('Importancia de Variables - Random Forest')
plt.xlabel('Importancia')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Conclusión

# %% [markdown]
# - El modelo Random Forest mostró un desempeño excelente, con una precisión total del 100% y una capacidad de detección de días de riesgo meteorológico del 95%. Solo un caso fue clasificado erróneamente como día normal. Este resultado demuestra un equilibrio adecuado entre precisión y generalización, superior al árbol de decisión que había sobreajustado los datos.
# - El análisis de importancia de variables reveló que la temperatura máxima es el factor más determinante para predecir el riesgo meteorológico, seguida por la temperatura mínima. La precipitación tuvo una influencia mínima en el modelo, lo cual concuerda con el comportamiento climático de la estación de Cañete, donde los eventos de lluvia intensa son escasos.

# %% [markdown]
# ## Guardar el modelo

# %%
import joblib

# Guardar modelo entrenado
joblib.dump(modelo_rf, '../models/modelo_riesgo_meteorologico_rf.pkl')

# %% [markdown]
# ### Cargar el modelo y probar con nuevos datos

# %%
# Cargar modelo guardado
modelo_cargado = joblib.load('../models/modelo_riesgo_meteorologico_rf.pkl')

# Crear un nuevo día para predecir
nuevo_dia = pd.DataFrame([{
    'precipitacion': 2.0,
    'temp_max': 34.5,
    'temp_min': 25.0
}])

# Usar para predecir
prediccion = modelo_rf.predict(nuevo_dia)
print("Predicción:", prediccion[0])