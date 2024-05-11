import tkinter as tk
from tkinter import ttk
from pytrends.request import TrendReq

host_language = 'es-AR'
country_trend = 'argentina'

# Inicializar pytrends
pytrends = TrendReq(hl=host_language)


# Función para obtener y mostrar las tendencias actuales para Argentina
def mostrar_tendencias():
    tendencias = pytrends.trending_searches(pn=country_trend)
    # Limpiar la lista antes de mostrar los nuevos resultados
    for widget in frame_tendencias.winfo_children():
        widget.destroy()
    # Mostrar cada tendencia en la ventana
    for i, tendencia in enumerate(tendencias[0]):
        etiqueta = tk.Label(frame_tendencias, text=f"{tendencia}", bg='white', font=('Arial', 12))
        etiqueta.pack(anchor='w', pady=2)

# Función para buscar y mostrar las consultas relacionadas
def buscar_tendencias_relacionadas():
    palabra_clave = entrada_texto.get()  # Obtener el texto ingresado por el usuario
    if palabra_clave:
        pytrends.build_payload([palabra_clave], geo='AR', gprop='')
        data = pytrends.related_queries()
        resultados = data.get(palabra_clave, {}).get('top', [])
        
        # Limpiar la lista antes de mostrar los nuevos resultados
        for widget in frame_resultados.winfo_children():
            widget.destroy()
        
        if resultados is not None:
            # Mostrar cada tendencia relacionada en la ventana
            for i, fila in resultados.iterrows():
                etiqueta = tk.Label(frame_resultados, text=f"{fila['query']}", bg='white', font=('Arial', 10))
                etiqueta.pack(anchor='w', pady=2)

ventana = tk.Tk()
ventana.title('Tendencias de Búsqueda en ' + country_trend)
ventana.configure(bg='white')

panel_principal = tk.PanedWindow(ventana, orient=tk.HORIZONTAL, bg='grey', sashrelief=tk.RAISED, sashwidth=2)
panel_principal.pack(fill=tk.BOTH, expand=1)

frame_tendencias = tk.Frame(panel_principal, bg='white')
panel_principal.add(frame_tendencias, stretch="always")

frame_derecho = tk.Frame(panel_principal, bg='white')
panel_principal.add(frame_derecho, stretch="always")

frame_busqueda = tk.Frame(frame_derecho, bg='white')
frame_busqueda.pack(side='top', fill='x', padx=10, pady=10)

entrada_texto = tk.Entry(frame_busqueda, width=50)
entrada_texto.pack(side='left', padx=10)

boton_buscar = tk.Button(frame_busqueda, text='Buscar', command=buscar_tendencias_relacionadas)
boton_buscar.pack(side='left')

frame_resultados = tk.Frame(frame_derecho, bg='white')
frame_resultados.pack(fill='both', expand=True, padx=10, pady=10)

mostrar_tendencias()

ventana.mainloop()
