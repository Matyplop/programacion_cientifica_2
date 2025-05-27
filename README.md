# Análisis de Proyectos Nualart (Aplicación Streamlit)

Esta aplicación Streamlit realiza un análisis de datos de proyectos, incluyendo limpieza de datos, cálculos de indicadores de rendimiento mediante un modelo de objetos, y generación de graficos.

## 🖼️ Características Principales

* **Carga de Datos:** Carga un dataset de proyectos desde un archivo CSV.
* **Limpieza de Datos:** Aplica varios pasos de limpieza (manejo de nulos, detección y eliminación de outliers) utilizando la clase `DataCleaner`.
* **Análisis Orientado a Objetos:** Utiliza las clases `Registro`, `Proyecto`, `Area`, `Equipo` e `Indicadores` para calcular métricas de negocio (costos, desviaciones, eficiencia, rankings).
* **Visualización Dinámica:** Genera automáticamente gráficos recomendados (histogramas, boxplots, dispersión, heatmaps) para análisis univariado y bivariado utilizando la clase `GestorDeGraficos`.
* **Interfaz Interactiva:** Presenta los resultados en una aplicación web fácil de usar construida con Streamlit.

## 🛠️ Estructura del Proyecto

* `app.py`: Script principal de la aplicación Streamlit.
* `data_cleaner.py`: Contiene la clase `DataCleaner` para la limpieza de datos.
* `models.py`: Define las clases `Registro`, `Proyecto`, `Area`, `Equipo` e `Indicadores`.
* `visualizador_dinamico.py`: Contiene la clase `GestorDeGraficos` y funciones auxiliares para la generación de gráficos.
* `dataset_con_nulos_outliers.csv`: Dataset de ejemplo utilizado por la aplicación (o el nombre de tu archivo).
* `requirements.txt`: Lista de dependencias de Python.
* `tests/` (Opcional, para pruebas): Carpeta donde residirán tus archivos de prueba.



## 🚀 Instalación y Configuración

1.  **Clonar el repositorio:**
    ```bash
    git clone (https://github.com/Matyplop/programacion_cientifica_2.git)
    
    ```



3.  **Instalar las dependencias:**
   
    Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
    Las dependencias clave son:
    * streamlit
    * pandas
    * numpy
    * matplotlib
    * seaborn

4.  **Dataset:**
    `dataset_con_nulos_outliers.csv` 

## ▶️ Cómo Ejecutar la Aplicación

Desde la carpeta raíz de del proyecto se ejecuta:

```bash
streamlit run app.py
