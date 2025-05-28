import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def _identificar_tipo_variable(series: pd.Series) -> str:
    """
    Identifica el tipo de variable de una Serie de Pandas para guiar la selección de gráficos.


    series (pd.Series): La serie de Pandas a analizar.

    str: Una cadena que describe el tipo de variable:
             'numérica', 'categórica (numérica)', 'fecha/hora',
             'categórica', 'categórica (objeto)', o 'mixta/desconocida'.
    """
    if pd.api.types.is_numeric_dtype(series):
        # Si es numérica pero tiene pocos valores únicos y no es float, se trata como categórica.
        if series.nunique() < 10 and series.dtype not in ['float64', 'float32']:
            return 'categórica (numérica)'
        return 'numérica'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'fecha/hora'
    elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return 'categórica'
    elif series.nunique() < 20: # Si es de tipo objeto pero tiene pocos valores únicos
        return 'categórica (objeto)'
    else:
        return 'mixta/desconocida'

def _crear_directorio_si_no_existe(path: str) -> None:
    """
    Crea un directorio en la ruta especificada si no existe.

    path (str): La ruta del directorio a crear.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def _plot_barra(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str = None, titulo: str = "", es_conteo: bool = False) -> None:
    """
    Genera un gráfico de barras

    Si `es_conteo` es True, genera un countplot de `x_col`.
    Si `es_conteo` es False, genera un barplot del promedio de `y_col` agrupado por `x_col`.
    Muestra solo las 15 categorías principales por frecuencia (conteo) o promedio (barplot).

        ax (plt.Axes): El eje de Matplotlib donde se dibujará el gráfico.
        df (pd.DataFrame): El DataFrame que contiene los datos.
        x_col (str): El nombre de la columna para el eje X (categorías).
        y_col: El nombre de la columna para el eje Y (valores numéricos para promediar).
                               Requerido si `es_conteo` es False. Defaults to None.
        es_conteo: Si True, realiza un conteo de `x_col`. Si False,
                                    calcula el promedio de `y_col` para cada `x_col`. Defaults to False.
    """
    if es_conteo:
        # Ordena por frecuencia y toma las 15 categorías más frecuentes
        order = df[x_col].value_counts().index[:15]
        sns.countplot(x=x_col, data=df, ax=ax, palette="viridis", order=order)
        ax.set_title(titulo, fontsize=9)
    else:
        if y_col is None or not pd.api.types.is_numeric_dtype(df[y_col]):
            ax.text(0.5, 0.5, f"Columna Y ('{y_col or 'N/A'}')\nno es numérica o no se proporcionó.", ha='center', va='center', color='red', fontsize=8, wrap=True)
            ax.set_title(f"Error: {titulo}", fontsize=9)
            return
        # Agrupa por x_col, calcula la media de y_col, ordena y toma las 15 principales
        order = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).index[:15]
        sns.barplot(x=x_col, y=y_col, data=df, ax=ax, palette="viridis", estimator=np.mean, errorbar=None, order=order)
        ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='x', rotation=30, ha='right', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

def _plot_dispersion(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, titulo: str = "") -> None:
    """
    Genera un gráfico de dispersión en el Eje (Axes) proporcionado.
    Si el DataFrame tiene más de 500 filas, se toma una muestra aleatoria de 500 filas.

        ax (plt.Axes): El eje de Matplotlib donde se dibujará el gráfico.
        df (pd.DataFrame): El DataFrame que contiene los datos.
        x_col (str): El nombre de la columna para el eje X.
        y_col (str): El nombre de la columna para el eje Y.
    """
    # Muestreo para evitar gráficos muy densos si hay muchos datos
    sample_df = df.sample(n=min(500, len(df)), random_state=1) if len(df) > 500 else df
    sns.scatterplot(x=x_col, y=y_col, data=sample_df, ax=ax, alpha=0.6, edgecolor="w", s=20)
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7)

def _plot_histograma(ax: plt.Axes, df: pd.DataFrame, col: str, titulo: str = "", bins: int = 15) -> None:
    """
    Genera un histograma con una estimación de densidad kernel (KDE) en el Eje (Axes) proporcionado.

        ax (plt.Axes): El eje de Matplotlib donde se dibujará el gráfico.
        df (pd.DataFrame): El DataFrame que contiene los datos.
        col (str): El nombre de la columna para la cual se generará el histograma.
        bins: Número de bins para el histograma. Defaults to 15.
    """
    sns.histplot(df[col], ax=ax, kde=True, bins=bins, color="skyblue", edgecolor="black")
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

def _plot_boxplot(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str = None, titulo: str = "") -> None:
    """
    Genera un gráfico de caja (boxplot) en el Eje (Axes) proporcionado.

    Si `y_col` es None, genera un boxplot univariado para `x_col`.
    Si `y_col` se proporciona, genera boxplots de `y_col` para cada categoría en `x_col`.
    En el caso bivariado, muestra solo las 15 categorías principales de `x_col`
    ordenadas por la mediana de `y_col`.

        ax (plt.Axes): El eje de Matplotlib donde se dibujará el gráfico.
        df (pd.DataFrame): El DataFrame que contiene los datos.
        x_col (str): El nombre de la columna para el eje X.
        y_col: El nombre de la columna para el eje Y.
    """
    if y_col is None: # Boxplot univariado
        sns.boxplot(y=x_col, data=df, ax=ax, color="lightblue", width=0.4)
        ax.set_title(titulo, fontsize=9)
    else: # Boxplot bivariado
        # Ordena por mediana y toma las 15 categorías más relevantes
        order = df.groupby(x_col)[y_col].median().sort_values(ascending=False).index[:15]
        sns.boxplot(x=x_col, y=y_col, data=df, ax=ax, palette="pastel", width=0.5, order=order)
        ax.set_title(titulo, fontsize=9)
        ax.tick_params(axis='x', rotation=30, ha='right', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

def _plot_area(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, titulo: str = "") -> None:
    """
    Genera un gráfico de área en el Eje proporcionado.
    Los datos se ordenan por `x_col` antes de graficar. Si `x_col` es de tipo fecha/hora,
    se convierte a cadena para el `fill_between`.

        ax (plt.Axes): El eje de Matplotlib donde se dibujará el gráfico.
        df (pd.DataFrame): El DataFrame que contiene los datos.
        x_col (str): El nombre de la columna para el eje X.
        y_col (str): El nombre de la columna para el eje Y.
    """
    df_sorted = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_sorted[x_col]):
        df_sorted = df_sorted.sort_values(by=x_col)
    else: # Intenta ordenar para otros tipos, maneja error si no es ordenable
        try:
            df_sorted = df_sorted.sort_values(by=x_col)
        except TypeError:
            ax.text(0.5,0.5, f"No se puede ordenar '{x_col}'\npara gráfico de área.", ha='center', va='center', color='red', fontsize=8, wrap=True)
            ax.set_title(f"Error: {titulo}", fontsize=9)
            return


    x_data_for_fill = df_sorted[x_col].astype(str) if pd.api.types.is_datetime64_any_dtype(df_sorted[x_col]) else df_sorted[x_col]

    ax.fill_between(x_data_for_fill, df_sorted[y_col], alpha=0.4, color="cornflowerblue")
    sns.lineplot(x=x_col, y=y_col, data=df_sorted, ax=ax, marker='', color='darkblue', lw=1.5) # lineplot usa x_col original
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='x', rotation=30, labelsize=7) # Rotar si x_data_for_fill es string largo
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7)



def _plot_linea(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, titulo: str = "") -> None:
    """
    Genera un gráfico de línea en el Eje proporcionado.
    Si `x_col` es de tipo fecha/hora o numérico y monotónicamente creciente, se ordena por `x_col`.

        ax (plt.Axes): El eje de Matplotlib donde se dibujará el gráfico.
        df (pd.DataFrame): El DataFrame que contiene los datos.
        x_col (str): El nombre de la columna para el eje X.
        y_col (str): El nombre de la columna para el eje Y.
    """
    df_sorted = df.copy()
    # Ordenar si x_col es fecha/hora o si es un ID numérico creciente.
    if pd.api.types.is_datetime64_any_dtype(df_sorted[x_col]) or \
       (pd.api.types.is_numeric_dtype(df_sorted[x_col]) and df_sorted[x_col].is_monotonic_increasing):
        df_sorted = df_sorted.sort_values(by=x_col)

    sns.lineplot(x=x_col, y=y_col, data=df_sorted, ax=ax, marker='o', markersize=3, color="teal", lw=1)
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='x', rotation=30, ha='right', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7)

def _plot_heatmap_correlacion(ax: plt.Axes, df: pd.DataFrame, titulo: str = "") -> None:
    """
    Genera un heatmap de la matriz de correlación de las columnas numéricas del DataFrame.

    Args:
        ax (plt.Axes): El eje de Matplotlib donde se dibujará el gráfico.
        df (pd.DataFrame): El DataFrame que contiene los datos.
    """
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.shape[1] < 2: # Necesitas al menos dos columnas numéricas
        ax.text(0.5, 0.5, "No hay suficientes\ncolumnas numéricas\npara un heatmap.",
                ha='center', va='center', fontsize=8, color='gray', wrap=True)
        ax.set_title(titulo + " (Datos Insuficientes)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    corr_matrix = df_numeric.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=.3, ax=ax, cbar=True, annot_kws={"size": 6})
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='x', labelsize=7, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=7, rotation=0)


class GestorDeGraficos:
    """
    Clase para gestionar la generación de visualizaciones automáticas
    basadas en los tipos de datos de las columnas de un DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el GestorDeGraficos.

        df (pd.DataFrame): El DataFrame de Pandas para el cual se generarán gráficos.

        ValueError: Si `df` no es un DataFrame de Pandas.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("La entrada 'df' debe ser un DataFrame de Pandas.")
        self.df = df
       

    def _obtener_recomendaciones(self, col1_name: str, col2_name: str = None) -> list:
    
        if col1_name not in self.df.columns or (col2_name and col2_name not in self.df.columns):
            print(f"Advertencia: Una o ambas columnas ('{col1_name}', '{col2_name}') no se encuentran en el DataFrame.")
            return []
        tipo_col1 = _identificar_tipo_variable(self.df[col1_name])
        tipo_col2 = _identificar_tipo_variable(self.df[col2_name]) if col2_name else None
        recomendaciones_config = []

        if col2_name is None: # Análisis Univariado
            if tipo_col1 == 'numérica':
                recomendaciones_config.append({"plot_type": "histograma", "kwargs": {"col": col1_name}, "descripcion": f"Distribución de {col1_name}"})
                recomendaciones_config.append({"plot_type": "boxplot_univariado", "kwargs": {"x_col": col1_name}, "descripcion": f"Boxplot de {col1_name}"})
            elif tipo_col1.startswith('categórica'): # Incluye 'categórica (numérica)' y 'categórica (objeto)'
                recomendaciones_config.append({"plot_type": "barra_conteo", "kwargs": {"x_col": col1_name}, "descripcion": f"Conteo de {col1_name}"})
                if self.df[col1_name].nunique() <= 6: 
                    recomendaciones_config.append({"plot_type": "pastel", "kwargs": {"col": col1_name}, "descripcion": f"Proporciones de {col1_name}"})

        else: # Análisis Bivariado
        
            c1, c2, t1, t2 = col1_name, col2_name, tipo_col1, tipo_col2
            if tipo_col1.startswith('numérica') and tipo_col2.startswith('categórica'):
                c1, c2, t1, t2 = col2_name, col1_name, tipo_col2, tipo_col1 # c1 siempre será categórica, c2 numérica

            if t1 == 'numérica' and t2 == 'numérica':
                recomendaciones_config.append({"plot_type": "dispersion", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Dispersión: {c2} vs {c1}"})
             
                if t1 == 'fecha/hora' or \
                   ('id' in c1.lower() and hasattr(self.df[c1], 'is_monotonic_increasing') and self.df[c1].is_monotonic_increasing):
                    recomendaciones_config.append({"plot_type": "linea", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Tendencia: {c2} vs {c1}"})

            elif t1.startswith('categórica') and t2 == 'numérica':
                recomendaciones_config.append({"plot_type": "barra_promedio", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Promedio de {c2} por {c1}"})
                recomendaciones_config.append({"plot_type": "boxplot_bivariado", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Distribución de {c2} por {c1}"})

            elif (t1 == 'fecha/hora' and t2 == 'numérica'): # Asegurando que x_col sea la fecha
                 recomendaciones_config.append({"plot_type": "linea", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Serie Temporal: {c2} vs {c1}"})
          
            elif t1.startswith('categórica') and t2.startswith('categórica'):
               
                pass
        return recomendaciones_config

    def generar_visualizaciones(self, col1_name: str, col2_name: str = None,
                                export_dir: str = "graficos_exportados", show_plot: bool = True,
                                base_figsize_w: float = 3.5,
                                base_figsize_h: float = 2.8
                               ) -> tuple:
       
        recomendaciones = self._obtener_recomendaciones(col1_name, col2_name)

        if not recomendaciones:
            print(f"No se generaron recomendaciones para '{col1_name}'" + (f" y '{col2_name}'" if col2_name else "") + ".")
            return None, [], []

        num_recomendaciones = len(recomendaciones)

        if num_recomendaciones == 1: ncols, nrows = 1, 1
        elif num_recomendaciones == 2: ncols, nrows = 2, 1
        elif num_recomendaciones == 3: ncols, nrows = 2, 2 # Preferir 2x2 para 3 plots, dejando uno vacío
        elif num_recomendaciones == 4: ncols, nrows = 2, 2
        else:
            ncols = 3
            nrows = (num_recomendaciones + ncols - 1) // ncols # 

        fig_width = base_figsize_w * ncols
        fig_height = base_figsize_h * nrows

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)
        axes_flat = axes.flatten() # Facilita iterar sobre los ejes
        file_paths = []

        for i, config in enumerate(recomendaciones):
            if i >= len(axes_flat): break
            ax = axes_flat[i]
            plot_type = config["plot_type"]
            plot_kwargs = config["kwargs"]
            plot_descripcion = config["descripcion"]

            try:
                
                if plot_type == "histograma":
                    _plot_histograma(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "boxplot_univariado":
                    _plot_boxplot(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "barra_conteo":
                    _plot_barra(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion, es_conteo=True)
                elif plot_type == "pastel":
                    _plot_pastel(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion) # Asume _plot_pastel definida
                elif plot_type == "dispersion":
                    _plot_dispersion(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "linea":
                    _plot_linea(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion) # Asume _plot_linea definida
                elif plot_type == "barra_promedio":
                    _plot_barra(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion, es_conteo=False)
                elif plot_type == "boxplot_bivariado":
                    _plot_boxplot(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "heatmap_correlacion": # Asume _plot_heatmap_correlacion definida
                    _plot_heatmap_correlacion(ax=ax, df=self.df.copy(), titulo=plot_descripcion)
              
                else:
                    ax.text(0.5, 0.5, f"Tipo de gráfico '{plot_type}'\nno implementado o no reconocido.", ha='center', va='center', color='orange', fontsize=8, wrap=True)
                    ax.set_title(f'{plot_descripcion} (No Implementado)', fontsize=9)

            except Exception as e:
                print(f"Error al generar gráfico '{plot_descripcion}' con tipo '{plot_type}': {e}")
                ax.text(0.5, 0.5, f'Error al generar:\n{plot_type}', ha='center', va='center', color='red', fontsize=8, wrap=True)
                ax.set_title(f'{plot_descripcion} (Error)', fontsize=9)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout(pad=0.7, h_pad=1.2 if nrows > 1 else 0.7, w_pad=0.7)

        if export_dir:
            _crear_directorio_si_no_existe(export_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"analisis_{col1_name.replace(' ', '_').replace('/', '_')}"
            if col2_name: base_filename += f"_vs_{col2_name.replace(' ', '_').replace('/', '_')}"
            full_figure_filename = f"{base_filename}_{timestamp}.png"
            full_figure_path = os.path.join(export_dir, full_figure_filename)
            try:
                fig.savefig(full_figure_path, dpi=150, bbox_inches='tight')
                file_paths.append(full_figure_path)
                print(f"Figura guardada en: {full_figure_path}")
            except Exception as e:
                print(f"Error al guardar la figura en '{full_figure_path}': {e}")

        if show_plot:
            plt.show()

        return fig, recomendaciones, file_paths

def generar_graficos_dinamicos(df: pd.DataFrame, col1_name: str, col2_name: str = None,
                               export_dir: str = None,
                               show_plot: bool = False,
                               base_figsize_w: float = 3.5,
                               base_figsize_h: float = 2.8
                              ) -> tuple:
 
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Se requiere un DataFrame de Pandas no vacío.")
        return None, [], []

    manager = GestorDeGraficos(df)
    return manager.generar_visualizaciones(col1_name, col2_name, export_dir, show_plot, base_figsize_w, base_figsize_h)
