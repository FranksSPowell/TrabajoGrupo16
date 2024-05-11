# m_recomendador.py

import pickle
import pandas as pd

def f_cargar_modelo():
    with open('modelo_entrenado.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return modelo

def f_generar_recomendaciones(perfil_usuario, n_recomendaciones):
    modelo = f_cargar_modelo()
    perfil_array = perfil_usuario.values  # Convertir el DataFrame en un array

    if perfil_array.size == 0:
        print("Perfil de usuario está vacío, no se pueden generar recomendaciones.")
        return {}

    probabilidades = modelo.predict_proba(perfil_array)

    if probabilidades.size == 0 or len(probabilidades[0]) == 0:
        print("No se generaron probabilidades, revisar el modelo y los datos de entrada.")
        return {}

    secciones_recomendadas_codigos = sorted(range(len(probabilidades[0])), key=lambda i: probabilidades[0][i], reverse=True)[:n_recomendaciones]
    
    seccion_original = pd.read_csv('secciones_originales.csv')
    secciones_recomendadas = seccion_original[seccion_original['codigo'].isin(secciones_recomendadas_codigos)]
    secciones_recomendadas = secciones_recomendadas.set_index('codigo')['seccion'].to_dict()

    probabilidades_secciones = [probabilidades[0][codigo] for codigo in secciones_recomendadas_codigos]

    secciones_recomendadas_con_probabilidades = {}
    for codigo, seccion, probabilidad in zip(secciones_recomendadas.keys(), secciones_recomendadas.values(), probabilidades_secciones):
        secciones_recomendadas_con_probabilidades[codigo] = {'seccion': seccion, 'probabilidad': probabilidad}
    
    return secciones_recomendadas_con_probabilidades