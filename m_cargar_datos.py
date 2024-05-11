# m_cargar_datos.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from m_config import URL
import os
import requests


def f_cargar_datos():

    # Cargar el dataframe
    filename="url_merge_SC.csv"
    if not os.path.exists(filename):
        # Si no existe, descarga el archivo de la URL
        response = requests.get(URL)
        with open(filename, 'wb') as file:
            file.write(response.content)
        historial_usuario = pd.read_csv(filename)
    else:
        # Si el archivo existe, lo carga directamente
        historial_usuario = pd.read_csv(filename)


    # Convertir las fechas a tipo datetime
    historial_usuario['fecha'] = pd.to_datetime(historial_usuario['fecha'])

    # Eliminamos 'pais' ya que consideramos que un usuario desde otro país, no brinda diferencias.
    historial_usuario = historial_usuario.drop(columns=['pais'])

    # Codificar las columnas categóricas
    label_encoder = LabelEncoder()
    historial_usuario['origen'] = label_encoder.fit_transform(historial_usuario['origen'])
    historial_usuario['genero'] = label_encoder.fit_transform(historial_usuario['genero'])
    historial_usuario['seccion'] = label_encoder.fit_transform(historial_usuario['seccion'])

    # Recupero los valores originales de 'seccion'
    secciones_originales = label_encoder.inverse_transform(historial_usuario['seccion'])

    # Obtener los valores únicos de 'seccion' después de ser codificados y guardarlos en 'secciones_originales.csv'
    codigos_seccion = historial_usuario['seccion'].unique()
    valores_originales = label_encoder.inverse_transform(codigos_seccion)
    seccion_original = pd.DataFrame({'codigo': codigos_seccion, 'seccion': valores_originales})
    seccion_original.sort_values(by='codigo', inplace=True)
    seccion_original.reset_index(drop=True, inplace=True)

    # Calculamos características adicionales
    frecuencia_lectura_usuario = historial_usuario.groupby('usuario').size().reset_index(name='frecuencia_lectura')  # Frecuencia de lectura por usuario
    total_notas_leidas_usuario = historial_usuario.groupby('usuario')['notas_UxFxS'].sum().reset_index(name='total_notas_leidas')  # Cantidad total de notas leídas por usuario
    interacciones_usuario_seccion = historial_usuario.groupby(['usuario', 'seccion']).size().reset_index(name='interacciones_usuario_seccion')  # Interacciones entre usuarios y secciones
    patrones_lectura_temporales = historial_usuario.groupby(['usuario', historial_usuario['fecha'].dt.dayofweek])['notas_UxFxS'].sum().unstack(fill_value=0).add_prefix('lecturas_por_dias_')
    promedio_notas_por_usuario_seccion = historial_usuario.groupby(['usuario', 'seccion'])['notas_UxFxS'].mean().reset_index()  # Agrupar por usuario y sección, y calcular la media de notas leídas

    # Renombrar la columna para mayor claridad
    promedio_notas_por_usuario_seccion = promedio_notas_por_usuario_seccion.rename(columns={'notas_UxFxS': 'promedio_notas'})

    # Agregar las nuevas columnas calculadas al dataset original
    historial_usuario = historial_usuario.merge(frecuencia_lectura_usuario, on='usuario')
    historial_usuario = historial_usuario.merge(total_notas_leidas_usuario, on='usuario')
    historial_usuario = historial_usuario.merge(interacciones_usuario_seccion, on=['usuario', 'seccion'])
    historial_usuario = historial_usuario.merge(promedio_notas_por_usuario_seccion, on=['usuario', 'seccion'], how='left')
    historial_usuario = historial_usuario.merge(patrones_lectura_temporales, on='usuario')

    # Obtener el último día en el dataset
    ultimo_dia = historial_usuario['fecha'].max()

    # Obtener la última fecha para cada usuario
    ultimas_fechas = historial_usuario.groupby('usuario')['fecha'].max().reset_index()
    ultimas_fechas.rename(columns={'fecha': 'ultima_fecha'}, inplace=True)
    
    # Unir las últimas fechas al dataframe original
    historial_usuario = historial_usuario.merge(ultimas_fechas, on='usuario')    

    # Separar en entrenamiento y prueba, filtrando los datos para dejar fuera el último día
    train = historial_usuario[historial_usuario['fecha'] != ultimo_dia]
    test = historial_usuario[historial_usuario['fecha'] == ultimo_dia]

    # Dropear columnas 'usuario', 'ultima_fecha', 'fecha'
    train_data = train.drop(columns=['usuario', 'ultima_fecha', 'fecha'])
    test_data = test.drop(columns=['usuario', 'ultima_fecha', 'fecha'])

    # Guarda todas las variables en archivos csv
    historial_usuario.to_csv('historial_usuario.csv', index=False)
    seccion_original.to_csv('secciones_originales.csv', index=False)
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)

    return historial_usuario, secciones_originales, train_data, test_data, train, test