# Sistema de Recomendación de Contenidos para Diario Digital

Este proyecto implementa un sistema de recomendación avanzado que utiliza técnicas de aprendizaje automático y análisis de datos para personalizar y mejorar la experiencia de lectura de los usuarios en un diario digital.

## Funcionalidades

- **Recuperación de Datos:** Extrae información de navegación y preferencias de usuarios desde una base de datos.
- **Procesamiento de Datos:** Prepara y procesa los datos para el análisis y modelado.
- **Generación de Recomendaciones:** Utiliza modelos como XGBoost y técnicas de filtrado colaborativo para generar recomendaciones personalizadas.
- **Evaluación de Recomendaciones:** Mide la eficacia de las recomendaciones a través de métricas como precisión y recall.

## Estructura del Proyecto

## Módulos del Proyecto

### Entrenador del Modelo
- **`m_main_entrenamiento`**: Script principal para el proceso de entrenamiento del modelo.
- **`m_carga_datos`**: Carga y prepara los datos para el entrenamiento.
- **`m_entrenar_modelo`**: Entrena el modelo de machine learning utilizando XGBoost.
- **`m_evaluar_modelo`**: Evalúa el rendimiento del modelo entrenado.

### Accuracy del Modelo
- **`m_main_accuracy`**: Calcula la precisión global del modelo.
- **`m_recomendador_accuracy`**: Específico para evaluar la precisión del módulo de recomendación.

### Recomendador
- **`m_main_recomendacion.py`**: Controlador principal para las recomendaciones.
- **`m_crear_perfil_random.py`**: Genera perfiles de usuario aleatorios para pruebas.
- **`m_recomendador.py`**: Implementa la lógica de recomendación basada en los perfiles de usuario.

### Configuración
- **`m_config`**: Contiene las configuraciones utilizadas por todos los módulos del sistema.

### Analizador de Variables
- **`analisis_variables_relevantes`**: Analiza qué variables son más importantes para el modelo.

### Optimizador de Hiperparámetros
- **`optimizador_hiperparametros`**: Ajusta los hiperparámetros del modelo para optimizar su rendimiento.

├── scripts/
│ ├── m_main_entrenamiento.py
│ ├── m_carga_datos.py
│ ├── m_entrenar_modelo.py
│ ├── m_evaluar_modelo.py
│ ├── m_main_accuracy.py
│ ├── m_recomendador_accuracy.py
│ ├── m_main_recomendacion.py
│ ├── m_crear_perfil_random.py
│ ├── m_recomendador.py
│ ├── m_config.py
│ ├── analisis_variables_relevantes.py
│ └── optimizador_hiperparametros.py
├── scripts/
  ├──SEO_Trends.py

Archivo PKL
https://drive.google.com/drive/folders/1GyJZ7m6oZS7Nu_WE_AQfjqDd7G929j26?usp=sharing


## Uso SEO

Para ejecutar el sistema de SEO:

1. Instala las dependencias con `pip install -r requirements.txt`.
2. Configura las variables necesarias
3. Ejecuta los scripts correspondientes.

## Uso RECOMENDADOR

Para ejecutar el sistema de SEO:

1. Configura las variables necesarias
2. Ejecuta los scripts correspondientes.

## Contribución

Para contribuir al proyecto, por favor haz fork del repositorio, crea una rama, aplica tus cambios y envía un pull request para su revisión.




