import os
import re
import pandas as pd
import folium
import streamlit as st
from streamlit.components.v1 import html

# --------- FUNCIONES ---------

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

def generar_dataframe_desde_archivos(ruta_carpeta):
    archivos = os.listdir(ruta_carpeta)
    datos = []
    for i, archivo in enumerate(archivos, start=1):
        if not archivo.lower().endswith(".data"):
            continue
        ubicacion, lat, lon = extraer_datos(archivo)
        try:
            lat = float(lat)
            lon = float(lon)
            if ubicacion and lat and lon:
                nombre_completo = f"Archivo {i} - {ubicacion}"
                datos.append({'Ubicación': nombre_completo, 'Latitud': lat, 'Longitud': lon})
        except (TypeError, ValueError):
            print(f"[Advertencia] No se pudo procesar: {archivo}")
            continue
    return pd.DataFrame(datos).drop_duplicates()

def generar_mapa(df, zoom_start=5, map_style="OpenStreetMap"):
    lat_centro = df['Latitud'].mean()
    lon_centro = df['Longitud'].mean()
    
    # Crear el mapa con el centro y el zoom inicial configurado
    mapa = folium.Map(
        location=[lat_centro, lon_centro],
        zoom_start=zoom_start,  # Ajuste del nivel de zoom inicial
        tiles=map_style,
        control_scale=True      # Activar el control de escala
    )

    # Agregar marcadores
    for _, row in df.iterrows():
        popup_html = f"""
        <b>{row['Ubicación']}</b><br>
        """
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=popup_html,
            tooltip=row['Ubicación']
        ).add_to(mapa)
    
    return mapa

# --------- APP STREAMLIT ---------

st.set_page_config(page_title="Mapa")

st.title("🌦️ Mapa de Estaciones Meteorológicas")
st.caption("Seleccionadas automáticamente desde la carpeta `DatosIdeamProcesados`")

# Carpeta de origen
carpeta_datos = os.path.join(os.path.dirname(__file__), "DatosIdeamProcesados")
df_mapa = generar_dataframe_desde_archivos(carpeta_datos)

if df_mapa.empty:
    st.warning("No se encontraron estaciones válidas.")
else:
    st.success(f"{len(df_mapa)} estaciones encontradas.")
    mapa = generar_mapa(df_mapa)
    mapa_html = mapa._repr_html_()

    # Mostrar mapa en Streamlit
    html(mapa_html, height=900)