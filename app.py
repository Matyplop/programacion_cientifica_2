import streamlit as st
import pandas as pd
import numpy as np # Necesario para algunas operaciones de Pandas/Seaborn
import os
import matplotlib.pyplot as plt 
from datetime import datetime

# Importar tus módulos personalizados
from data_cleaner import DataCleaner
from models import Registro, Proyecto, Area
from visualizador_dinamico import generar_graficos_dinamicos

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Análisis Nualart (Directo)")

# --- Título ---
st.title("Análisis de Proyectos Nualart")
st.caption(f"Generado: {datetime.now().strftime('%A, %d de %B de %Y, %H:%M')} - Temuco, Chile")
st.markdown("---")

# --- PASO 0: Datos Originales (Vista Previa) ---
st.header("0. Datos Originales")
default_file_path = "dataset_con_nulos_outliers.csv" 
df_original = None
if os.path.exists(default_file_path):
    try:
        df_original = pd.read_csv(default_file_path)
        st.dataframe(df_original.head(), height=180)
        
        # Mostrar resumen de nulos del original en un expander
        null_summary_original = df_original.isnull().sum()
        null_summary_original = null_summary_original[null_summary_original > 0]
        if not null_summary_original.empty:
            with st.expander("Ver Resumen de Valores Nulos en Datos Originales", expanded=False):
                st.dataframe(null_summary_original.to_frame(name='Cantidad de Nulos'), height=150)
        else:
            st.info("El dataset original no presenta valores nulos.")
            
    except Exception as e:
        st.error(f"Error al cargar datos originales: {e}")
        st.stop() 
else:
    st.error(f"Archivo '{default_file_path}' no encontrado. La aplicación no puede continuar.")
    st.stop()
st.markdown("---")

# --- PASO 1: Proceso de Limpieza de Datos ---
st.header("1. Proceso y Resultados de Limpieza de Datos")

@st.cache_data 
def ejecutar_limpieza_completa_con_log(filepath):
    try:
        data_handler = DataCleaner(filepath) 
        limpieza_log_estructurado = [] # Para el log en la UI
        outliers_detectados_ui = {}   # Para mostrar los DFs de outliers

        # Log: Tipos de datos iniciales
        initial_dtypes_df = data_handler.cleaned_data.dtypes.reset_index()
        initial_dtypes_df.columns = ['Columna', 'Tipo de Dato Original']
        initial_dtypes_df['Tipo de Dato Original'] = initial_dtypes_df['Tipo de Dato Original'].astype(str)
        limpieza_log_estructurado.append({
            "paso": "Tipos de Datos Iniciales",
            "detalle_df": initial_dtypes_df
        })

        # Log: Detección de outliers
        cols_para_detectar_outliers = ["costo_real", "cantidad_trabajadores"]
        for col_detect in cols_para_detectar_outliers:
            if col_detect in data_handler.cleaned_data.columns and pd.api.types.is_numeric_dtype(data_handler.cleaned_data[col_detect]):
                df_outliers = data_handler.detect_outliers_iqr(col_detect) 
                summary_text = f"Detección de outliers en `{col_detect}` usando IQR. Función: `detect_outliers_iqr('{col_detect}')`."
                if not df_outliers.empty:
                    outliers_detectados_ui[f"Outliers Detectados en '{col_detect}' ({len(df_outliers)} filas)"] = df_outliers
                    summary_text += f" Se encontraron {len(df_outliers)} outliers (ver tabla abajo)."
                else:
                    summary_text += " No se detectaron outliers significativos."
                limpieza_log_estructurado.append({"paso": "Detección de Outliers", "columna": col_detect, "resumen": summary_text, "funcion": f"detect_outliers_iqr('{col_detect}')"})

        # Log: Eliminación de outliers
        cols_para_eliminar_outliers = ["costo_real", "cantidad_trabajadores"]
        for col_outlier in cols_para_eliminar_outliers:
            if col_outlier in data_handler.cleaned_data.columns and pd.api.types.is_numeric_dtype(data_handler.cleaned_data[col_outlier]):
                filas_antes = data_handler.cleaned_data.shape[0]
                data_handler.remove_outliers_iqr(col_outlier) 
                filas_despues = data_handler.cleaned_data.shape[0]
                filas_eliminadas = filas_antes - filas_despues
                limpieza_log_estructurado.append({
                    "paso": "Eliminación de Outliers", "columna": col_outlier,
                    "metric_label": f"Filas Eliminadas ('{col_outlier}')", "metric_value": filas_eliminadas,
                    "detalle": f"Dataset pasó de {filas_antes} a {filas_despues} filas.",
                    "funcion": f"remove_outliers_iqr('{col_outlier}')"
                })
        
        # Log: Rellenar nulos con mediana
        cols_a_rellenar_mediana = [
            "cantidad_trabajadores", "costo_real", "costo_estimado",
            "avance_estimado", "avance_real"
        ]
        for col_median in cols_a_rellenar_mediana:
            if col_median in data_handler.cleaned_data.columns:
                if pd.api.types.is_numeric_dtype(data_handler.cleaned_data[col_median]):
                    if data_handler.cleaned_data[col_median].isnull().any():
                        nulos_antes = data_handler.cleaned_data[col_median].isnull().sum()
                        mediana_usada = data_handler.cleaned_data[col_median].median() 
                        data_handler.rellenar_con_mediana(col_median) 
                        limpieza_log_estructurado.append({
                            "paso": "Imputación de Nulos con Mediana", "columna": col_median,
                            "metric_label": f"Nulos Rellenados ('{col_median}')", "metric_value": nulos_antes,
                            "detalle": f"Se imputaron {nulos_antes} nulos con la mediana: {mediana_usada:,.2f}.",
                            "funcion": f"rellenar_con_mediana('{col_median}')"
                        })
                    else:
                        limpieza_log_estructurado.append({"paso": "Imputación de Nulos con Mediana", "columna": col_median, "detalle": f"No se encontraron valores nulos para rellenar en `{col_median}`."})
                else:
                    limpieza_log_estructurado.append({"paso": "Imputación de Nulos con Mediana", "columna": col_median, "detalle": f"Columna `{col_median}` no es numérica."})
            else:
                 limpieza_log_estructurado.append({"paso": "Imputación de Nulos con Mediana", "columna": col_median, "detalle": f"Columna `{col_median}` no encontrada."})
        
        return data_handler.cleaned_data, limpieza_log_estructurado, outliers_detectados_ui
    except Exception as e:
        print(f"Error crítico durante la limpieza: {e}") 
        return None, [], {} 

dataframe_final_limpio, log_limpieza_detallado, outliers_info_ui = ejecutar_limpieza_completa_con_log(default_file_path)

# Mostrar el log de limpieza estructurado en un expander
with st.expander("Ver Detalles del Proceso de Limpieza Aplicado", expanded=False):
    if log_limpieza_detallado:
        for item in log_limpieza_detallado:
            st.markdown(f"**{item['paso']}**" + (f" (Columna: `{item.get('columna','N/A')}`)" if "columna" in item else ""))
            if "metric_label" in item and "metric_value" in item: 
                st.metric(label=item["metric_label"], value=item["metric_value"])
            if "detalle" in item: st.write(item["detalle"])
            if "resumen" in item: st.write(item["resumen"])
            if "funcion" in item: st.caption(f"Función DataCleaner: `{item['funcion']}`")
            if "detalle_df" in item: st.dataframe(item["detalle_df"], height=200)
            st.markdown("---") 
    else:
        st.info("No se registraron pasos de limpieza detallados.")
    
    if outliers_info_ui:
        st.markdown("**DataFrames de Outliers Detectados (antes de su eliminación):**")
        for titulo_outlier, df_o in outliers_info_ui.items():
            st.caption(titulo_outlier)
            st.dataframe(df_o, height=150)

st.subheader("Datos Limpios (Resultado)") # Título más claro
if dataframe_final_limpio is not None and not dataframe_final_limpio.empty:
    st.dataframe(dataframe_final_limpio.head(), height=180)
    st.success(f"Limpieza completada. Filas: {len(dataframe_final_limpio)} (Originales: {len(df_original)})")
else:
    st.error("El DataFrame está vacío o no se pudo limpiar. Revisa la consola para detalles.")
    st.stop()
st.markdown("---")

# --- PASO 2: Resumen Textual del Análisis POO ---
st.header("2. Resumen del Análisis Orientado a Objetos")
registros_list = []
proyectos_dict = {}
areas_dict = {}

required_cols_registro = ["id", "proyecto", "area", "equipo", "costo_estimado", "costo_real", "avance_estimado", "avance_real", "cantidad_trabajadores"]
if not all(col in dataframe_final_limpio.columns for col in required_cols_registro):
    st.error(f"Faltan columnas críticas para el análisis POO: {', '.join(required_cols_registro)}. No se puede continuar con esta sección.")
else:
    for _, row in dataframe_final_limpio.iterrows():
        try:
            registros_list.append(Registro(
                id=row["id"], proyecto=row["proyecto"], area=row["area"], equipo=row["equipo"],
                costo_estimado=row["costo_estimado"], costo_real=row["costo_real"],
                avance_estimado=row["avance_estimado"], avance_real=row["avance_real"],
                trabajadores=row["cantidad_trabajadores"]
            ))
        except Exception: pass 

    for r_obj in registros_list:
        if r_obj.proyecto:
            if r_obj.proyecto not in proyectos_dict:
                proyectos_dict[r_obj.proyecto] = Proyecto(r_obj.proyecto)
            proyectos_dict[r_obj.proyecto].agregar_registro(r_obj)
    for r_obj in registros_list:
        if r_obj.area:
            if r_obj.area not in areas_dict:
                areas_dict[r_obj.area] = Area(r_obj.area)
            areas_dict[r_obj.area].agregar_registro(r_obj)

    col_poo_proy, col_poo_area = st.columns(2)
    with col_poo_proy:
        st.subheader("Análisis por Proyecto")
        if proyectos_dict:
            for nombre_proyecto, proyecto_obj in sorted(proyectos_dict.items()):
                if not proyecto_obj.registros: continue
                st.markdown(f"**📌 {nombre_proyecto}**")
                st.markdown(f"&nbsp;&nbsp;- Costo Est.: ${proyecto_obj.costo_total_estimado():,.0f}")
                st.markdown(f"&nbsp;&nbsp;- Costo Real: ${proyecto_obj.costo_total_real():,.0f}")
                st.markdown(f"&nbsp;&nbsp;- Desviación: ${proyecto_obj.desviacion_presupuesto():,.0f}")
                st.markdown(f"&nbsp;&nbsp;- Rendimiento: {proyecto_obj.rendimiento_promedio():.2f}%")
        else: st.info("No hay datos de proyectos para resumir.")
    with col_poo_area:
        st.subheader("Análisis por Área")
        if areas_dict:
            for nombre_area, area_obj in sorted(areas_dict.items()):
                if not area_obj.registros: continue
                st.markdown(f"**📍 {nombre_area}**")
                st.markdown(f"&nbsp;&nbsp;- Eficiencia: {area_obj.eficiencia_promedio():.2f}%")
                st.markdown(f"&nbsp;&nbsp;- Sobrecosto: ${area_obj.total_sobrecosto():,.0f}")
        else: st.info("No hay datos de áreas para resumir.")
st.markdown("---")

# --- PASO 3: Visualizaciones Automáticas Recomendadas ---
st.header("3. Panel de Visualizaciones Recomendadas")

analisis_a_realizar = []
columnas_univariadas_interes = ['costo_real', 'costo_estimado', 'avance_real', 'avance_estimado', 'cantidad_trabajadores'] 
for col_name in columnas_univariadas_interes:
    if col_name in dataframe_final_limpio.columns:
        analisis_a_realizar.append({"col1": col_name, "col2": None, "titulo_seccion": f"'{col_name}'"})

pares_bivariados_interes = [
    ('costo_estimado', 'costo_real'), ('avance_estimado', 'avance_real'),
    ('cantidad_trabajadores', 'costo_real'), ('cantidad_trabajadores', 'avance_real')
]
for col1, col2 in pares_bivariados_interes:
    if col1 in dataframe_final_limpio.columns and col2 in dataframe_final_limpio.columns:
        analisis_a_realizar.append({"col1": col1, "col2": col2, "titulo_seccion": f"'{col1}' vs '{col2}'"})

base_fig_w_dinamico = 2.8 
base_fig_h_dinamico = 2.2 

st_cols_viz = st.columns(2) 
col_idx_viz_streamlit = 0

# No se usa st.spinner aquí para mantener el código más corto, pero se podría añadir
for analisis_info in analisis_a_realizar:
    with st_cols_viz[col_idx_viz_streamlit % 2]: 
        st.subheader(f"Análisis para: {analisis_info['titulo_seccion']}")
        
        resultado_graficos = generar_graficos_dinamicos(
            df=dataframe_final_limpio,
            col1_name=analisis_info["col1"],
            col2_name=analisis_info["col2"],
            export_dir=None, 
            show_plot=False,
            base_figsize_w=base_fig_w_dinamico, 
            base_figsize_h=base_fig_h_dinamico
        )
        
        figura_grupo = None
        if resultado_graficos and isinstance(resultado_graficos, tuple) and len(resultado_graficos) > 0:
            figura_grupo = resultado_graficos[0] 
            # El segundo elemento (lista de configs) no se usa aquí para interpretaciones individuales
        
        if figura_grupo:
            st.pyplot(figura_grupo)
            plt.close(figura_grupo) 
    col_idx_viz_streamlit += 1
st.markdown("---")

st.caption(f"Fin del reporte. Análisis Nualart - {datetime.now().strftime('%d/%m/%Y %H:%M')}")