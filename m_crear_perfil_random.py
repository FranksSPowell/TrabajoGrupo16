# m_crear_perfil_random.py

from m_cargar_datos import f_cargar_datos
import pandas as pd

def f_crear_perfil_random(usuario_random):

    #print("Cargando datos para crear el perfil de usuario aleatorio...")
    _, _, _, _, train, _ = f_cargar_datos()
    #usuario_random = train['usuario'].sample(n=1).values[0]
    #print("Usuario seleccionado:", usuario_random)

    # Recuperar las últimas secciones leídas por el usuario seleccionado
    ultimas_secciones_leidas = train[train['usuario'] == usuario_random].sort_values('fecha', ascending=False).head(5)['seccion'].values
    seccion_original = pd.read_csv('secciones_originales.csv')
    nombres_ultimas_secciones_leidas = seccion_original.loc[seccion_original['codigo'].isin(ultimas_secciones_leidas), 'seccion'].values
    
    #print("Últimas secciones leídas por el usuario:", usuario_random)
    #for seccion in nombres_ultimas_secciones_leidas:
    #    print(f"- {seccion}")
    #print()
    #print("Generando perfil de usuario...")
    perfil_usuario = train[train['usuario'] == usuario_random]
    perfil_usuario = perfil_usuario.drop(columns=['usuario', 'seccion', 'ultima_fecha', 'fecha'])
    #print()
    # Guardar el perfil del usuario en un archivo CSV 
    #perfil_usuario.to_csv('perfil_usuario.csv', index=False)
    #print("Perfil de usuario guardado en 'perfil_usuario.csv'.")
    
    return perfil_usuario, nombres_ultimas_secciones_leidas