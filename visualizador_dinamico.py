import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def _identificar_tipo_variable(series: pd.Series) -> str:

    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() < 10 and series.dtype not in ['float64', 'float32']:
            return 'categórica (numérica)'
        return 'numérica'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'fecha/hora'
    elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return 'categórica'
    elif series.nunique() < 20: 
        return 'categórica (objeto)'
    else:
        return 'mixta/desconocida'

def _crear_directorio_si_no_existe(path: str):
    """Crea un directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)



def _plot_barra(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str = None, titulo: str = "", es_conteo: bool = False):
    if es_conteo:
        order = df[x_col].value_counts().index[:15]
        sns.countplot(x=x_col, data=df, ax=ax, palette="viridis", order=order)
        ax.set_title(titulo, fontsize=9)
    else:
        if y_col is None or not pd.api.types.is_numeric_dtype(df[y_col]):
            ax.text(0.5, 0.5, f'{y_col or "Y"} no numérico', ha='center', va='center', color='red', fontsize=8)
            ax.set_title(f'Error: {y_col or "Y"} no numérico', fontsize=9)
            return
        order = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).index[:15]
        sns.barplot(x=x_col, y=y_col, data=df, ax=ax, palette="viridis", estimator=np.mean, errorbar=None, order=order)
        ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='x', rotation=30, ha='right', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

def _plot_dispersion(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, titulo: str = ""):
    sample_df = df.sample(n=min(500, len(df)), random_state=1) if len(df) > 500 else df
    sns.scatterplot(x=x_col, y=y_col, data=sample_df, ax=ax, alpha=0.6, edgecolor="w", s=20)
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7)

def _plot_histograma(ax: plt.Axes, df: pd.DataFrame, col: str, titulo: str = "", bins=15):
    sns.histplot(df[col], ax=ax, kde=True, bins=bins, color="skyblue", edgecolor="black")
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')


def _plot_boxplot(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str = None, titulo: str = ""):
    if y_col is None: 
        sns.boxplot(y=x_col, data=df, ax=ax, color="lightblue", width=0.4) 
        ax.set_title(titulo, fontsize=9)
    else: 
        order = df.groupby(x_col)[y_col].median().sort_values(ascending=False).index[:15]
        sns.boxplot(x=x_col, y=y_col, data=df, ax=ax, palette="pastel", width=0.5, order=order) 
        ax.set_title(titulo, fontsize=9)
        ax.tick_params(axis='x', rotation=30, ha='right', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')


def _plot_area(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, titulo: str = ""):
    df_sorted = df.copy() 
    if pd.api.types.is_datetime64_any_dtype(df_sorted[x_col]):
        df_sorted = df_sorted.sort_values(by=x_col)
    else:
        try:
            df_sorted = df_sorted.sort_values(by=x_col)
        except TypeError:
            ax.text(0.5,0.5, f"No se puede ordenar {x_col}\npara gráfico de área", ha='center', va='center', color='red', fontsize=8)
            return

    x_data_for_fill = df_sorted[x_col].astype(str) if pd.api.types.is_datetime64_any_dtype(df_sorted[x_col]) else df_sorted[x_col]
    ax.fill_between(x_data_for_fill, df_sorted[y_col], alpha=0.4, color="cornflowerblue")
    sns.lineplot(x=x_col, y=y_col, data=df_sorted, ax=ax, marker='', color='darkblue', lw=1.5)
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.7)


class GestorDeGraficos:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("La entrada 'df' debe ser un DataFrame de Pandas.")
        self.df = df
        # No se usa plot_catalog, las llamadas serán directas o con if/elif

    def _obtener_recomendaciones(self, col1_name: str, col2_name: str = None) -> list:
        # Esta función se mantiene igual que en la versión anterior (id="visualizador_dinamico_py_fix")
        # Devuelve una lista de diccionarios como:
        # {"plot_type": "histograma", "kwargs": {"col": col1_name}, "descripcion": f"Distribución de {col1_name}"}
        if col1_name not in self.df.columns or (col2_name and col2_name not in self.df.columns):
            return []
        tipo_col1 = _identificar_tipo_variable(self.df[col1_name])
        tipo_col2 = _identificar_tipo_variable(self.df[col2_name]) if col2_name else None
        recomendaciones_config = []

        if col2_name is None: # Análisis Univariado
            if tipo_col1 == 'numérica':
                recomendaciones_config.append({"plot_type": "histograma", "kwargs": {"col": col1_name}, "descripcion": f"Distribución de {col1_name}"})
                recomendaciones_config.append({"plot_type": "boxplot_univariado", "kwargs": {"x_col": col1_name}, "descripcion": f"Boxplot de {col1_name}"})
            elif tipo_col1.startswith('categórica'):
                recomendaciones_config.append({"plot_type": "barra_conteo", "kwargs": {"x_col": col1_name}, "descripcion": f"Conteo de {col1_name}"})
                if self.df[col1_name].nunique() <= 6:
                     recomendaciones_config.append({"plot_type": "pastel", "kwargs": {"col": col1_name}, "descripcion": f"Proporciones de {col1_name}"})
        
        else: # Análisis Bivariado
            c1, c2, t1, t2 = col1_name, col2_name, tipo_col1, tipo_col2
            if tipo_col1.startswith('numérica') and tipo_col2.startswith('categórica'):
                c1, c2, t1, t2 = col2_name, col1_name, tipo_col2, tipo_col1 

            if t1 == 'numérica' and t2 == 'numérica':
                recomendaciones_config.append({"plot_type": "dispersion", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Dispersión: {c2} vs {c1}"})
                if t1 == 'fecha/hora' or ('id' in c1.lower() and self.df[c1].is_monotonic_increasing if hasattr(self.df[c1], 'is_monotonic_increasing') else False):
                     recomendaciones_config.append({"plot_type": "linea", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Tendencia: {c2} vs {c1}"})
            
            elif t1.startswith('categórica') and t2 == 'numérica':
                recomendaciones_config.append({"plot_type": "barra_promedio", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Promedio de {c2} por {c1}"})
                recomendaciones_config.append({"plot_type": "boxplot_bivariado", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Distribución de {c2} por {c1}"})
            
            elif (t1 == 'fecha/hora' and t2 == 'numérica'):
                recomendaciones_config.append({"plot_type": "linea", "kwargs": {"x_col": c1, "y_col": c2}, "descripcion": f"Serie Temporal: {c2} vs {c1}"})
            
            elif t1.startswith('categórica') and t2.startswith('categórica'):
                 pass 
        return recomendaciones_config

    def generar_visualizaciones(self, col1_name: str, col2_name: str = None, 
                                export_dir: str = "graficos_exportados", show_plot: bool = True,
                                base_figsize_w: float = 3.5, 
                                base_figsize_h: float = 2.8
                               ):
        recomendaciones = self._obtener_recomendaciones(col1_name, col2_name)
        
        if not recomendaciones:
            return None, [], [] 

        num_recomendaciones = len(recomendaciones)
        
        if num_recomendaciones == 1: ncols, nrows = 1, 1
        elif num_recomendaciones == 2: ncols, nrows = 2, 1
        elif num_recomendaciones == 3: ncols, nrows = 2, 2 # Preferir 2x2
        elif num_recomendaciones == 4: ncols, nrows = 2, 2
        else: 
            ncols = 3 
            nrows = (num_recomendaciones + ncols - 1) // ncols
        
        fig_width = base_figsize_w * ncols
        fig_height = base_figsize_h * nrows
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)
        axes_flat = axes.flatten()
        file_paths = []
        
        for i, config in enumerate(recomendaciones):
            if i >= len(axes_flat): break 
            ax = axes_flat[i]
            plot_type = config["plot_type"]
            plot_kwargs = config["kwargs"]
            plot_descripcion = config["descripcion"]
            
            try:
                # Dispatcher explícito if/elif
                if plot_type == "histograma":
                    _plot_histograma(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "boxplot_univariado":
                    _plot_boxplot(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "barra_conteo":
                    _plot_barra(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion, es_conteo=True)
                elif plot_type == "pastel":
                    _plot_pastel(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "dispersion":
                    _plot_dispersion(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "linea":
                    _plot_linea(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "barra_promedio":
                    _plot_barra(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion, es_conteo=False)
                elif plot_type == "boxplot_bivariado":
                    _plot_boxplot(ax=ax, df=self.df.copy(), **plot_kwargs, titulo=plot_descripcion)
                elif plot_type == "heatmap_correlacion": # Nueva condición
                    _plot_heatmap_correlacion(ax=ax, df=self.df.copy(), titulo=plot_descripcion)
                # Añadir aquí más elif para "violin" o "area" si se reincorporan a _obtener_recomendaciones
                else:
                    ax.text(0.5, 0.5, f"Tipo '{plot_type}'\nno implementado", ha='center', va='center', color='red', fontsize=8)
                    ax.set_title(f'{plot_descripcion} (Error)', fontsize=9)

            except Exception as e:
                print(f"Error al generar gráfico '{plot_descripcion}' con tipo '{plot_type}': {e}")
                ax.text(0.5, 0.5, 'Error al generar', ha='center', va='center', color='red', fontsize=8)
                ax.set_title(f'{plot_descripcion} (Error)', fontsize=9)

        for i in range(num_recomendaciones, len(axes_flat)): 
            axes_flat[i].set_visible(False)
        
        plt.tight_layout(pad=0.7, h_pad=1.2 if nrows > 1 else 0.7, w_pad=0.7)

        if export_dir: 
            _crear_directorio_si_no_existe(export_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"analisis_{col1_name.replace(' ', '_')}"
            if col2_name: base_filename += f"_vs_{col2_name.replace(' ', '_')}"
            full_figure_filename = f"{base_filename}_{timestamp}.png"
            full_figure_path = os.path.join(export_dir, full_figure_filename)
            try:
                fig.savefig(full_figure_path, dpi=150, bbox_inches='tight') 
                file_paths.append(full_figure_path)
            except Exception as e:
                print(f"Error al guardar la figura: {e}")
        
        if show_plot: plt.show() 
            
        return fig, recomendaciones, file_paths 

def generar_graficos_dinamicos(df: pd.DataFrame, col1_name: str, col2_name: str = None, 
                              export_dir: str = None, 
                              show_plot: bool = False, 
                              base_figsize_w: float = 3.5, 
                              base_figsize_h: float = 2.8 
                              ):
    if not isinstance(df, pd.DataFrame) or df.empty: 
        return None, [], [] 
            
    manager = GestorDeGraficos(df)
    return manager.generar_visualizaciones(col1_name, col2_name, export_dir, show_plot, base_figsize_w, base_figsize_h)

def _plot_heatmap_correlacion(ax: plt.Axes, df: pd.DataFrame, titulo: str = ""):
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.shape[1] < 2: # Necesitas al menos dos columnas numéricas
        ax.text(0.5, 0.5, "No hay suficientes\ncolumnas numéricas\npara un heatmap.",
                ha='center', va='center', fontsize=8, color='gray')
        ax.set_title(titulo + " (Datos Insuficientes)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    corr_matrix = df_numeric.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=.3, ax=ax, cbar=True, annot_kws={"size": 6}) # Ajustar tamaño de anotación
    ax.set_title(titulo, fontsize=9)
    ax.tick_params(axis='both', labelsize=7, rotation=45) # Ajustar ticks para mejor visualización