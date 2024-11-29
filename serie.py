import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Cargar los datos
def cargar_datos(ruta):
    try:
        datos = pd.read_excel(ruta)
        print("Archivo cargado correctamente.")
        return datos
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

# Preprocesar los datos
def preprocesar_datos(datos):
    try:
        # Codificar columna categórica 'género'
        if 'género' in datos.columns:
            encoder = OneHotEncoder()
            datos_codificados = encoder.fit_transform(datos[['género']]).toarray()
        else:
            print("Columna 'género' no encontrada. Omitiendo codificación.")
            datos_codificados = np.array([])

        # Normalizar columnas numéricas
        columnas_numericas = ['Duración (min)', 'Año Estreno', 'Calificación IMDb', 'Número de Temporadas']
        columnas_disponibles = [col for col in columnas_numericas if col in datos.columns]

        if columnas_disponibles:
            scaler = MinMaxScaler()
            datos_normalizados = scaler.fit_transform(datos[columnas_disponibles])
        else:
            print("No se encontraron columnas numéricas para normalizar.")
            datos_normalizados = np.array([])

        # Concatenar los datos preprocesados
        if datos_codificados.size > 0 and datos_normalizados.size > 0:
            datos_finales = np.hstack((datos_codificados, datos_normalizados))
        elif datos_codificados.size > 0:
            datos_finales = datos_codificados
        else:
            datos_finales = datos_normalizados

        return datos_finales
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        return None

# Calcular similitud de coseno
def calcular_similitud(datos_preprocesados):
    return cosine_similarity(datos_preprocesados)

# Recomendar series similares
def recomendar_series(similitudes, datos, nombre_serie, top_n=3):
    try:
        if nombre_serie not in datos['Serie'].values:
            print(f"La serie '{nombre_serie}' no está en la lista.")
            return []

        indice_serie = datos.index[datos['Serie'] == nombre_serie][0]
        indices_similares = np.argsort(similitudes[indice_serie])[-(top_n+1):-1][::-1]
        series_similares = datos.iloc[indices_similares]['Serie'].values

        return series_similares
    except Exception as e:
        print(f"Error al recomendar series: {e}")
        return []

# Ruta del archivo
ruta_archivo = "100_Series_Nombradas.xlsx"

# Proceso principal
datos = cargar_datos(ruta_archivo)

if datos is not None:
    datos_preprocesados = preprocesar_datos(datos)
    if datos_preprocesados is not None:
        similitudes = calcular_similitud(datos_preprocesados)
        # Probar recomendaciones
        series_a_probar = ["Breaking Bad", "Stranger Things", "The Office"]
        for serie in series_a_probar:
            recomendaciones = recomendar_series(similitudes, datos, serie)
            print(f"Recomendaciones para '{serie}': {recomendaciones}")
