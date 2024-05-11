import pandas as pd
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from m_crear_perfil_random import  f_crear_perfil_random
from m_recomendador import f_cargar_modelo, f_generar_recomendaciones


def obtener_top_secciones(df, usuario_id, top_n=6):
    n=30
    # Filtrar el dataframe por usuario
    df_usuario = df[df['usuario'] == usuario_id].copy()
    
    # Asegurarse de que la columna 'fecha' está en formato de fecha
    df_usuario['fecha'] = pd.to_datetime(df_usuario['fecha'])
    
    # Ordenar el DataFrame por fecha de forma descendente
    df_usuario = df_usuario.sort_values(by='fecha', ascending=False)
    
    # Obtener los días únicos y limitar a los últimos n días activos
    unique_days = df_usuario['fecha'].dt.date.unique()[:n]
    
    # Filtrar las filas que caen dentro de los últimos n días activos
    df_usuario = df_usuario[df_usuario['fecha'].dt.date.isin(unique_days)]
    
    # Contar las ocurrencias de cada sección
    conteo_secciones = df_usuario['seccion'].value_counts()
    
    # Obtener las top_n secciones
    top_secciones = conteo_secciones.head(top_n).index.tolist()
    
    return top_secciones

def f_main_recomendacion(usuario_random,k):
    #print(usuario_random)
    n_recomendaciones = k  

    perfil_usuario, ultimas_secciones_leidas = f_crear_perfil_random(usuario_random)
    # Cargar modelo entrenado
    #modelo = f_cargar_modelo()
    # Generar recomendaciones para el perfil de usuario
    secciones_recomendadas = f_generar_recomendaciones(perfil_usuario, n_recomendaciones)  
    # Mostrar las secciones recomendadas con sus nombres originales y probabilidades
    secciones = [info['seccion'] for info in secciones_recomendadas.values()]
    #print(secciones)

    return secciones


def ultima_fecha_lectura(df, fecha_col='fecha'):
    try:
        df[fecha_col] = pd.to_datetime(df[fecha_col], format='%Y-%m-%d')
    except Exception as e:
        print(f"Error al convertir las fechas: {e}")
        return None, None
    
    # Obtener la última fecha para cada usuario
    ultimas_fechas = df.groupby('usuario')[fecha_col].max().reset_index()
    ultimas_fechas.rename(columns={fecha_col: 'ultima_fecha'}, inplace=True)
    
    # Unir las últimas fechas al dataframe original
    df = df.merge(ultimas_fechas, on='usuario')
    
    # Separar en entrenamiento y prueba
    df_prueba = df[df[fecha_col] == df['ultima_fecha']]
    df_entrenamiento = df[df[fecha_col] != df['ultima_fecha']]
    
    # Dropear columna 'ultima_fecha'
    df_prueba = df_prueba.drop(columns=['ultima_fecha'])
    df_entrenamiento = df_entrenamiento.drop(columns=['ultima_fecha'])
    
    return df_entrenamiento, df_prueba

def prepare_data(df_navegacion):
    # Filtrar y limpiar datos antes de aplicar get_dummies
    df_navegacion = df_navegacion[['usuario', 'seccion', 'origen', 'pais']] 

    columns_to_encode = ['seccion', 'origen', 'pais']

    # Aplicar one-hot encoding solo a las columnas relevantes
    navegacion_encoded = pd.get_dummies(df_navegacion,  columns=columns_to_encode)

    # Agrupar por usuario y sumar
    perfil_usuario_navegacion = navegacion_encoded.groupby('usuario').sum().reset_index()

    # Normalizar las columnas one-hot por el total de acciones para cada prefijo por usuario
    for prefix in columns_to_encode:  # Añade o elimina prefijos según sea necesario
        # Obtener las columnas para cada prefijo
        columns = [col for col in perfil_usuario_navegacion if col.startswith(prefix)]
        # Calcular el total de acciones por usuario para cada grupo de columnas
        total_actions_per_group = perfil_usuario_navegacion[columns].sum(axis=1)
        # Normalizar cada grupo de columnas
        perfil_usuario_navegacion[columns] = perfil_usuario_navegacion[columns].div(total_actions_per_group, axis=0)

    return perfil_usuario_navegacion

def recommend_based_on_similarity(user_id, df_prueba, user_item_df, similarity_matrix, num_recommendations=6, day=None):
    if user_id not in user_item_df.index:
        return []

    # Ajustar por día de la semana si está disponible
    if day:
        df_prueba = df_prueba[df_prueba['dia_semana'] == day]

    similarities = similarity_matrix.loc[user_id]
    similar_users = similarities.sort_values(ascending=False).drop(user_id).index[:10]
    similar_users_sections = df_prueba[df_prueba['usuario'].isin(similar_users)]

    # Ponderar las secciones por recencia y relevancia
    recent_sections = similar_users_sections.groupby('seccion').agg({'fecha': 'max'}).sort_values(by='fecha', ascending=False)
    top_sections = recent_sections.head(num_recommendations).index.tolist()

    return top_sections


def create_user_item_matrix(dataframe, user_col, item_cols):
    """ Crea una matriz usuario-item usando pandas get_dummies para one-hot encoding de las secciones especificadas. """
    df_dummies = pd.get_dummies(dataframe.set_index(user_col)[item_cols])
    user_item_matrix = df_dummies.groupby(level=0).max().clip(upper=1) 
    
    # Convertir booleanos a enteros
    user_item_matrix = user_item_matrix.astype(int)

    return csr_matrix(user_item_matrix.values), user_item_matrix



def calculate_similarity(user_item_df):
    """ Calcula la similitud coseno entre los perfiles de usuario. """
    similarity_matrix = cosine_similarity(user_item_df)
    return pd.DataFrame(similarity_matrix, index=user_item_df.index, columns=user_item_df.index)


def recommend_based_on_date(user_id, df_prueba, user_item_df, num_recommendations=6):
    """ Recomienda secciones basadas en la última fecha de lectura del usuario y las lecturas de otros usuarios ese día. """
    
    user_last_reading = df_prueba[df_prueba['usuario'] == user_id]
    if user_last_reading.empty:
        return []

    last_reading_date = user_last_reading['fecha'].iloc[0]
    other_users_same_date = df_prueba[(df_prueba['fecha'] == last_reading_date) & (df_prueba['usuario'] != user_id)]

    if not other_users_same_date.empty:
        # Recomendar las secciones más populares leídas ese día por otros usuarios
        recommended_sections = other_users_same_date['seccion'].value_counts().head(num_recommendations).index.tolist()
    else:
        # Si no hay otros usuarios, recomienda las secciones más leídas por el usuario a partir de sus datos históricos
        user_sections = user_item_df.loc[user_id]
        recommended_sections = user_sections.sort_values(ascending=False).head(num_recommendations).index.tolist()
        #print(f"Recomendaciones basadas en las lecturas históricas del usuario {user_id}.")

    return recommended_sections


def combinar_recomendaciones(similar, xgboost, entrenamiento, max_recomendaciones=6):
    # Crear un conjunto de recomendaciones únicas
    recomendaciones_finales = set()

    # Contar la frecuencia de cada sección en todas las recomendaciones
    frecuencias = {}
    
    # Lista que contiene todos los sets de recomendaciones
    listas_recomendaciones = [set(similar), set(xgboost), set(entrenamiento)]
    
    # Contar las frecuencias de cada sección en las tres listas
    for lista in listas_recomendaciones:
        for seccion in lista:
            if seccion in frecuencias:
                frecuencias[seccion] += 1
            else:
                frecuencias[seccion] = 1
    
    # Ordenar las secciones por frecuencia descendente
    recomendaciones_ordenadas = sorted(frecuencias, key=frecuencias.get, reverse=True)
    
    # Seleccionar las recomendaciones con frecuencia mayor o igual a 2
    recomendaciones_finales.update([sec for sec in recomendaciones_ordenadas if frecuencias[sec] >= 2])
    
    # Si las recomendaciones finales no alcanzan el máximo deseado, añadir de XGBoost
    if len(recomendaciones_finales) < max_recomendaciones:
        # Filtro las secciones de XGBoost que no están en recomendaciones finales
        recomendaciones_adicionales = [sec for sec in xgboost if sec not in recomendaciones_finales]
        recomendaciones_finales.update(recomendaciones_adicionales[:max_recomendaciones - len(recomendaciones_finales)])
    
    return list(recomendaciones_finales)[:max_recomendaciones]

def evaluate_recommendations(df_prueba, df_entrenamiento, user_item_df, similarity_matrix, k):
    total_recall = 0
    total_precision = 0
    count = 0
    results = []
    """
    for index, row in df_prueba.iterrows():
        user_id = row['usuario']
    """
    #unique_users = df_prueba['usuario'].unique()[:20]
    unique_users = df_prueba['usuario'].unique()

    
    unique_users_count = len(df_prueba['usuario'].unique())
    n=0+1

    for user_id in unique_users:

        print(user_id," Usuaro:",n,"Pendientes de hacer:",n,"/",unique_users_count)
        n=n+1

        recomencacion_usuarios_similares = recommend_based_on_similarity(user_id, df_prueba, user_item_df, similarity_matrix, k)
        recomendaciones_xgboost = f_main_recomendacion(user_id, k)
        recomendaciones_historico = obtener_top_secciones(df_entrenamiento, user_id)

    
        #recomendaciones=recomencacion_usuarios_similares
        recomendaciones = combinar_recomendaciones(recomencacion_usuarios_similares, recomendaciones_xgboost, recomendaciones_historico)
        
        actual_sections = df_prueba[df_prueba['usuario'] == user_id]['seccion'].tolist()
        
        
    
        """
        if len(recomendaciones) < 6 or len(recomencacion_usuarios_similares) < 6 or len(recomendaciones_xgboost) < 6:
            print(recomencacion_usuarios_similares,"LEN: ",len(recomencacion_usuarios_similares))
            print(recomendaciones_xgboost,"LEN: ",len(recomendaciones_xgboost))
            print(recomendaciones_historico,"LEN: ",len(recomendaciones_historico))
            print("-----")
            print(recomendaciones,"LEN: ",len(recomendaciones))
            print("-----")
            print(actual_sections)
            print("No recomendó 6")
        """

       



        hits = sum(1 for section in recomendaciones if section in actual_sections)
        recall = hits / len(set(actual_sections)) if actual_sections else 0
        precision = hits / len(recomendaciones) if recomendaciones else 0

        total_recall += recall
        total_precision += precision
        count += 1

        results.append({
            'Usuario': user_id,
            'Recomendaciones': ', '.join(recomendaciones),
            'Lecturas Reales': ', '.join(actual_sections),
            'Aciertos': hits,
            'Recall': recall,
            'Precision': precision
        })

    mean_average_recall = total_recall / count if count > 0 else 0
    mean_average_precision = total_precision / count if count > 0 else 0

    return results, mean_average_recall, mean_average_precision



def calculate_mar(recommendations, actual_readings, k):
    user_recall = []
    for user_id in recommendations:
        # Cálculos para un usuario específico
        hits = sum(1 for rec in recommendations[user_id] if rec in actual_readings[user_id])
        user_recall.append(hits / len(set(actual_readings[user_id])))
        
    # Calcular el mAR
    mar_at_k = sum(user_recall) / len(user_recall)
    return mar_at_k


if __name__ == "__main__":

    url_usuarios =   'https://raw.githubusercontent.com/josberqui/TP-Final/main/Usuarios.csv'
    url_navegacion = 'https://raw.githubusercontent.com/josberqui/TP-Final/main/Navegacion.csv'

    df_usuarios = pd.read_csv(url_usuarios)
    df_navegacion = pd.read_csv(url_navegacion)


    usuarios_comunes = set(df_navegacion['usuario']).intersection(set(df_usuarios['usuario']))
    
    # Filtrar los dataframes para mantener solo los usuarios comunes
    df_usuarios = df_usuarios[df_usuarios['usuario'].isin(usuarios_comunes)]
    df_navegacion = df_navegacion[df_navegacion['usuario'].isin(usuarios_comunes)]

    df_entrenamiento, df_prueba = ultima_fecha_lectura(df_navegacion)
    df_prueba = df_prueba[['usuario', 'fecha', 'seccion']]

    perfil_usuario_navegacion = prepare_data(df_entrenamiento)
    df_usuarios_completo = pd.merge(df_usuarios, perfil_usuario_navegacion, on='usuario', how='left')
    df_usuarios_completo.fillna(0, inplace=True)
    item_cols = [col for col in df_usuarios_completo.columns if col not in ['usuario', 'fecha']]
    user_item_csr, user_item_df = create_user_item_matrix(df_usuarios_completo, 'usuario', item_cols)
    similarity_matrix = calculate_similarity(user_item_df)

    results, mar, map = evaluate_recommendations(df_prueba, df_entrenamiento, user_item_df, similarity_matrix, 6)

    print(f"Mean Average Recall: {mar:.2f}")
    print(f"Mean Average Precision: {map:.2f}")