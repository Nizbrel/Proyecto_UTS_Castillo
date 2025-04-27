Proyecto de Grado - Nicolás Hernán Castillo García

Optando por el Título de Tecnólogo en Electricidad Industrial

Unidades Tecnológicas de Santander

Bucaramanga, Santander

_________________________________________________________________________________
_________________________________________________________________________________



📌 Descripción del Proyecto

Este repositorio contiene una aplicación web desarrollada con Streamlit para el análisis del recurso eólico en Zonas de Alta Montaña de la Región Andina.

La app permite procesar datos históricos de velocidad del viento provenientes del IDEAM y otras fuentes como la NASA, aplicar ajustes estadísticos con la distribución de Weibull, y calcular el potencial de generación eólica en diferentes ubicaciones.

_________________________________________________________________________________
_________________________________________________________________________________


🚀 Funcionalidades Principales

📂 Carga de archivos con datos de velocidad del viento (desde este repositorio).

🧼 Limpieza y estandarización automática de los datos.

📈 Ajuste de la distribución de Weibull mediante diferentes métodos:

Método de Máxima Verosimilitud (ML)

✅ Verificación estadística del ajuste con la prueba de Kolmogorov-Smirnov.

⚡ Cálculo de potencia eólica disponible (W/m²).

🌎 Obtención automática de altitud vía coordenadas (API externa).

📊 Visualizaciones interactivas con Plotly (en versión web).

📥 Exportación de resultados a Excel (versión de escritorio).



_________________________________________________________________________________
_________________________________________________________________________________





📁 Estructura del Repositorio

📦 Proyecto_UTS_Castillo/

├── 📂 DatosIdeamProcesados/       # Archivos procesados desde IDEAM

├── 📂 DatosNasaProcesados/        # Archivos procesados desde NASA

├── 📜 versiongithubIDEAM.py       # Versión de análisis para datos IDEAM

├── 📜 versiongithubNASA.py        # Versión de análisis para datos NASA

├── 📜 VisualizadorMapa.py         # Visualizador de estaciones en mapa

├── 📜 CNE_IDEAM.xlsx              # Catastro Nacional de Estaciones del IDEAM

├── 📜 requirements.txt            # Dependencias necesarias

├── 📂 .devcontainer/              # Configuración opcional para entorno de desarrollo

└── 📄 README.md  



_________________________________________________________________________________
_________________________________________________________________________________




🛠️ Tecnologías Usadas
Python 3.11

Streamlit

Pandas / NumPy / Dask

SciPy / Statsmodels

Plotly

Open-Meteo API (altitud)

POWER LARC API (datos NASA)



_________________________________________________________________________________
_________________________________________________________________________________




🤝 Agradecimientos
A mi docente asesor Franky Cely por su acompañamiento y guía durante todas las etapas del proyecto.

Al semillero Alternativas de Generración de Energía (AGE) por el espacio de investigación y aprendizaje.

A las fuentes de datos como IDEAM y NASA por brindar acceso abierto a sus registros.

