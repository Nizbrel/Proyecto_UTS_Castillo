# La versión de la NASA (Código corregido)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import re
import folium
from io import BytesIO
from math import gamma as gamma_func  # Función gamma
import requests
import os

# Variable para definir el percentil de filtrado
PERCENTILE_THRESHOLD = 90  # Modifica este valor según sea necesario

# Función para cargar y limpiar datos usando chunks (se cachea la información)
@st.cache_data
def load_and_clean_data(file):
    chunks = pd.read_csv(file, names=["YEAR", "MO", "DY", "HR", "WS10M"], header=0, chunksize=100000)
    df_chunks = []
    for chunk in chunks:
        # Aseguramos que los campos de mes, día y hora tengan dos dígitos
        chunk['MO'] = chunk['MO'].astype(str).str.zfill(2)
        chunk['DY'] = chunk['DY'].astype(str).str.zfill(2)
        chunk['HR'] = chunk['HR'].astype(str).str.zfill(2)
        # Construimos la fecha en un solo campo
        chunk["Fecha"] = pd.to_datetime(
            chunk['YEAR'].astype(str) + '-' + chunk['MO'] + '-' + chunk['DY'] + ' ' + chunk['HR'] + ':00', 
            errors='coerce'
        )
        # Convertimos la velocidad y usamos el valor absoluto
        chunk["Velocidad"] = chunk["WS10M"].astype(float).abs()
        # Eliminamos valores nulos y filtramos velocidades mayores a 50 m/s
        df_chunks.append(chunk.dropna(subset=["Fecha", "Velocidad"]).query("Velocidad < 50"))
    df = pd.concat(df_chunks, ignore_index=True)
    return df

# Función para extraer información de ubicación desde el nombre del archivo usando expresiones regulares
def extraer_datos(nombre_archivo):
    patrones = [
        r'UBICACION\((.*?)\).*?LATITUD\(([-\d\.]+)\)LONGITUD\(([-\d\.]+)\)',
        r'_UBICACION_(.*?)_.*?_LATITUD_([-\d\.]+)_LONGITUD_([-\d\.]+)_',
        r'UBICACION\s*([\w\s\-\[\]]+).*?LATITUD_([-\d\.]+)_LONGITUD_([-\d\.]+)',
        r'_UBICACION_([\w\s\-\[\]]+)_LATITUD_([-\d\.]+)_LONGITUD_([-\d\.]+)'
    ]
    for patron in patrones:
        coincidencia = re.search(patron, nombre_archivo)
        if coincidencia:
            ubicacion, latitud, longitud = coincidencia.groups()
            return ubicacion.strip(), latitud.strip(), longitud.strip()
    return None, None, None

# Dataclass para almacenar los parámetros Weibull
@dataclass
class WeibullParams:
    k: float
    c: float

# Estimación por Máxima Verosimilitud de la distribución Weibull
def estimate_weibull_ml(data):
    params = weibull_min.fit(data, floc=0)
    return WeibullParams(params[0], params[2])

# Función para crear un mapa de Folium
def crear_mapa(lat, lon, ubicacion):
    mapa = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker(location=[lat, lon], popup=ubicacion, tooltip=ubicacion).add_to(mapa)
    mapa_io = BytesIO()
    mapa.save(mapa_io, close_file=False)
    mapa_io.seek(0)
    return mapa_io

# Título de la aplicación
st.set_page_config(page_title="NASA")
st.title("Comparativa Datos Nasa")

# Determinamos la ruta del directorio actual y la carpeta de datos
directorio_script = os.path.dirname(os.path.abspath(__file__))
carpeta_datos = os.path.join(directorio_script, "DatosNasaProcesados")


# Lista todos los archivos que terminan en .csv (sin distinguir mayúsculas/minúsculas)
archivos_data = [f for f in os.listdir(carpeta_datos) if f.lower().endswith(".csv")]

# Ordenamos los archivos alfabéticamente
archivos_data.sort()

# Extraemos la información de ubicación desde cada nombre de archivo
ubicaciones = []
for i, nombre_archivo in enumerate(archivos_data, start=1):
    ruta_archivo = os.path.join(carpeta_datos, nombre_archivo)
    ubicacion, lat_str, lon_str = extraer_datos(nombre_archivo)
    if ubicacion:
        ubicaciones.append((f"Archivo {i} - {ubicacion}", lat_str, lon_str, ruta_archivo))

# ------------------
# Selector con filtro de búsqueda
# ------------------
opciones = [u[0] for u in ubicaciones]
busqueda = st.text_input("🔍 Buscar ubicación")
opciones_filtradas = [op for op in opciones if busqueda.lower() in op.lower()] if busqueda else opciones

if opciones_filtradas:
    # Guarda la selección previa usando el estado de sesión
    if "selected_station" not in st.session_state:
        st.session_state.selected_station = opciones_filtradas[0]
    
    default_index = opciones_filtradas.index(st.session_state.selected_station) if st.session_state.selected_station in opciones_filtradas else 0
    
    ubicacion_seleccionada = st.selectbox(
        "📍 Selecciona la ubicación para el análisis",
        opciones_filtradas,
        index=default_index,
        key="select_ubicacion"
    )
    st.session_state.selected_station = ubicacion_seleccionada

    # Obtenemos los datos del archivo seleccionado
    seleccion = next(u for u in ubicaciones if u[0] == ubicacion_seleccionada)
    ruta_archivo_seleccionado = seleccion[3]
    
    # Conversión segura de latitud y longitud (se añade .strip() y manejo de errores)
    try:
        latitud = float(seleccion[1].strip())
        longitud = float(seleccion[2].strip())
    except ValueError as e:
        st.error(f"Error al convertir latitud/longitud en el archivo {seleccion[0]}: {e}")
        st.stop()
    
    # Carga y limpieza de los datos
    df = load_and_clean_data(ruta_archivo_seleccionado)
    if df.empty:
        st.error("Error: No hay datos válidos después de la limpieza.")
    else:
        # Estadísticas descriptivas
        stats = df["Velocidad"].describe()

        # Extraer municipio y departamento desde el nombre del archivo
        selected_file_name = os.path.basename(ruta_archivo_seleccionado)
        municipio, departamento = None, None
        patrones_municipio_departamento = [
            r'MUNICIPIO\((.*?)\).*?DEPARTAMENTO\((.*?)\)',
            r'_MUNICIPIO_(.*?)_.*?_DEPARTAMENTO_(.*?)_',
        ]
        for patron in patrones_municipio_departamento:
            coincidencia = re.search(patron, selected_file_name)
            if coincidencia:
                municipio, departamento = coincidencia.groups()
                break

        if municipio and departamento:
            st.markdown(f"**🏙️ Municipio:** {municipio}")
            st.markdown(f"**🗺️ Departamento:** {departamento}")
        else:
            st.warning("No se pudo extraer el municipio y el departamento.")

        # Formateo de las estadísticas para mostrarlas en tabla
        stats_rounded = stats.copy()
        stats_rounded["count"] = f"{int(stats_rounded['count'])}"
        for col in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            stats_rounded[col] = f"{stats_rounded[col]:.2f}"
        stats_with_units = stats_rounded.rename({
            "count": "conteo",
            "mean": "promedio (m/s)",
            "std": "std (m/s)",
            "min": "min (m/s)",
            "25%": "25% (m/s)",
            "50%": "50% (m/s)",
            "75%": "75% (m/s)",
            "max": "max (m/s)"
        })
        stats_df = pd.DataFrame(stats_with_units).reset_index()
        stats_df.columns = ["Estadística", "Valor"]
        st.markdown(
            """
            <style>
            .centered-table td, .centered-table th {
                text-align: center !important;
                vertical-align: middle !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.dataframe(stats_df.style.set_table_attributes('class="centered-table"'), use_container_width=True, hide_index=True)
        st.caption("Estadísticas Descriptivas Velocidad del Viento - Cuantitativa Continua")

        # Gráfica de la distribución de la velocidad del viento diaria
        st.markdown("##### Distribución de la Velocidad del Viento")
        df["Fecha2"] = df["Fecha"].dt.date
        df_daily = df.groupby("Fecha2")["Velocidad"].mean().reset_index()
        fig5 = px.scatter(
            df_daily, x="Fecha2", y="Velocidad",
            title="Velocidad del Viento Promedio Diario",
            labels={"Fecha2": "Fecha", "Velocidad": "Velocidad del Viento (m/s)"}
        )
        st.plotly_chart(fig5)

        # Filtrado por percentil para el análisis Weibull
        cutoff = np.percentile(df["Velocidad"], PERCENTILE_THRESHOLD)
        df_weibull = df[df["Velocidad"] <= cutoff]
        weibull_ml = estimate_weibull_ml(df_weibull["Velocidad"].to_numpy())

        # Creamos columnas para la visualización
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # Gráfica de CDF Empírica vs Weibull
        with col3:
            sorted_data = np.sort(df_weibull["Velocidad"])
            empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            weibull_cdf = weibull_min.cdf(sorted_data, weibull_ml.k, scale=weibull_ml.c)
            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(x=sorted_data, y=empirical_cdf, mode="lines", name="CDF Empírica"))
            fig_cdf.add_trace(go.Scatter(x=sorted_data, y=weibull_cdf, mode="lines", name="CDF Weibull", line=dict(dash="dash")))
            fig_cdf.update_layout(
                title="🔍 Comparación de CDF Empírica vs Weibull",
                xaxis_title="Velocidad (m/s)",
                yaxis_title="Probabilidad acumulada",
                template="plotly_white"
            )
            st.plotly_chart(fig_cdf)

        # Gráfica de histograma y PDF estimada de Weibull
        with col4:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=df_weibull["Velocidad"], nbinsx=10, histnorm="probability density",
                name="Datos", opacity=0.8, marker=dict(line=dict(width=1, color="black"))
            ))
            x_vals = np.linspace(df_weibull["Velocidad"].min(), df_weibull["Velocidad"].max(), 100)
            fig2.add_trace(go.Scatter(
                x=x_vals, y=weibull_min.pdf(x_vals, weibull_ml.k, scale=weibull_ml.c),
                mode="lines", name="Weibull ML", line=dict(dash="dash")
            ))
            fig2.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper", text=f"k = {weibull_ml.k:.4f} <br>c = {weibull_ml.c:.4f}", showarrow=False, font=dict(size=12, color="black"), bgcolor="white", bordercolor="black", borderwidth=1, borderpad=4, opacity=0.8, align="left")

            fig2.update_layout(
                title="📈 Distribución Weibull",
                xaxis_title="Velocidad (m/s)",
                yaxis_title="Densidad",
                barmode="overlay",
                template="plotly_white"
            )
            st.plotly_chart(fig2)

        # Mapa de ubicación geográfica
        with col1:
            st.markdown("<p style='font-family:Arial, sans-serif; font-size:15px; font-weight:bold; margin-top:30px;'>🌍 Ubicación Geográfica</p>", unsafe_allow_html=True)
            # Se vuelve a obtener la información para el mapa y se hace la conversión segura
            selected_location = next(u for u in ubicaciones if u[0] == ubicacion_seleccionada)
            try:
                lat_map = float(selected_location[1].strip())
                lon_map = float(selected_location[2].strip())
            except ValueError as e:
                st.error(f"Error al convertir latitud/longitud para el mapa: {e}")
                st.stop()
            # Se usa una verificación que permite coordenadas 0.0
            if ubicacion_seleccionada is not None and (lat_map is not None) and (lon_map is not None):
                mapa_io = crear_mapa(lat_map, lon_map, ubicacion_seleccionada)
                st.components.v1.html(mapa_io.getvalue().decode(), width=500, height=400)

        # Diagrama de caja por periodos de 5 años
        with col2:
            df_filtered = df[df["Velocidad"] <= np.percentile(df["Velocidad"], 97)].copy()  # copy para evitar SettingWithCopyWarning
            df_filtered["Año"] = df_filtered["Fecha"].dt.year
            df_filtered["Grupo 5 años"] = (df_filtered["Año"] // 5) * 5
            fig4 = px.box(
                df_filtered, x="Grupo 5 años", y="Velocidad",
                title="📊 Diagramas de Caja cada 5 años (Percentil 97)",
                labels={"Grupo 5 años": "Periodo (Años)", "Velocidad": "Velocidad del Viento (m/s)"}
            )
            st.plotly_chart(fig4)

        # Evaluación de la hipótesis nula
        st.write("**Hipótesis Nula (H₀):** Los datos siguen una distribución Weibull.")
        max_diff = np.max(np.abs(empirical_cdf - weibull_cdf))
        st.write(f"Máxima diferencia (D): {max_diff*100:.2f}%")
        if max_diff < 0.1:  # Umbral arbitrario para la decisión
            st.success("No se rechaza H₀: Los datos siguen una distribución Weibull.")
        else:
            st.error("Se rechaza H₀: Los datos no siguen una distribución Weibull.")

        # Sección para la estimación de potencia
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("#### Fórmula de Potencia por unidad de área (Weibull)")
            for _ in range(6):
                st.write("") 
            st.latex(r"\frac{P}{A} = \frac{1}{2} \rho c^3 \Gamma\left(1 + \frac{3}{k}\right)")
            for _ in range(6):
                st.write("") 
        with col6:
            st.markdown("#### Parámetros Weibull y Potencia estimada")
            # Función para calcular la densidad del aire a determinada altitud
            def calcular_densidad_altura(h):
                rho0 = 1.225  # kg/m³ a nivel del mar
                T0 = 288.15  # K
                g = 9.80665  # m/s²
                M = 0.0289644  # kg/mol
                R = 8.3144598  # J/(mol·K)
                return rho0 * (1 - 0.0065 * h / T0) ** ((g * M / (R * 0.0065)) - 1)

            # Función para obtener la altitud usando la API Open-Elevation
            def obtener_altura(lat, lon):
                try:
                    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        return data['results'][0]['elevation']
                except Exception as e:
                    st.error(f"Error al obtener la altitud: {e}")
                # Valor por defecto si hay error
                return 2000

            altura = obtener_altura(latitud, longitud)
            rho = calcular_densidad_altura(altura)
            gamma_val = gamma_func(1 + 3 / weibull_ml.k)
            potencia = 0.5 * rho * (weibull_ml.c ** 3) * gamma_val

            st.latex(f"k = {weibull_ml.k:.2f}")
            st.latex(f"c = {weibull_ml.c:.2f}")
            st.latex(f"\\Gamma\\left(1 + \\frac{{3}}{{k}}\\right) = {gamma_val:.2f}")
            st.latex(f"\\rho = {rho:.2f} \\ \\text{{kg/m}}^3")
            st.latex(r"\frac{P}{A} = " + "{:.2f}".format(potencia) + r"\ W/m^2")
