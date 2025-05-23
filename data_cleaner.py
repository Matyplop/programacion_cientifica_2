import numpy as np
import os
import unicodedata
import pandas as pd

class DataCleaner:
    def __init__(self, filepath_or_buffer):
        """
        Inicializa el objeto DataCleaner cargando un archivo CSV.
        Acepta una ruta de archivo o un objeto buffer (para Streamlit uploads).
        """
        if isinstance(filepath_or_buffer, str): # Si es una ruta de archivo
            if not os.path.exists(filepath_or_buffer):
                raise FileNotFoundError(f"No se encontró el archivo: {filepath_or_buffer}")
            self.data = pd.read_csv(filepath_or_buffer)
        else: # Asumir que es un buffer (ej. de st.file_uploader)
            self.data = pd.read_csv(filepath_or_buffer)
            
        self.cleaned_data = self.data.copy()
        # Los prints irán a la consola donde se ejecuta Streamlit
        print(f"Dataset cargado exitosamente con {self.cleaned_data.shape[0]} filas y {self.cleaned_data.shape[1]} columnas.")

    def report_nulls(self):
        null_report = self.cleaned_data.isnull().sum()
        print("Reporte de valores nulos por columna:")
        print(null_report[null_report > 0])
        return null_report

    def detect_outliers_iqr(self, column):
        if column not in self.cleaned_data.columns:
            print(f"Advertencia: La columna '{column}' no existe para detectar outliers.")
            return pd.DataFrame()
        if not pd.api.types.is_numeric_dtype(self.cleaned_data[column]):
            print(f"Advertencia: La columna '{column}' no es numérica. No se pueden detectar outliers con IQR.")
            return pd.DataFrame()

        Q1 = self.cleaned_data[column].quantile(0.25)
        Q3 = self.cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0: # Evitar problemas si todos los valores son iguales después del filtrado inicial
            print(f"Rango intercuartílico (IQR) es cero para la columna '{column}'. No se detectarán outliers por este método aquí.")
            return pd.DataFrame()

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = self.cleaned_data[(self.cleaned_data[column] < limite_inferior) | (self.cleaned_data[column] > limite_superior)]
        print(f"🔎 Se detectaron {outliers.shape[0]} outliers en la columna '{column}' usando IQR.")
        return outliers

    def remove_outliers_iqr(self, column):
        if column not in self.cleaned_data.columns:
            print(f"Advertencia: La columna '{column}' no existe para eliminar outliers.")
            return
        if not pd.api.types.is_numeric_dtype(self.cleaned_data[column]):
            print(f"Advertencia: La columna '{column}' no es numérica. No se pueden eliminar outliers con IQR.")
            return

        Q1 = self.cleaned_data[column].quantile(0.25)
        Q3 = self.cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            print(f"Rango intercuartílico (IQR) es cero para la columna '{column}'. No se eliminarán outliers por este método.")
            return

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        initial_shape = self.cleaned_data.shape
        self.cleaned_data = self.cleaned_data[(self.cleaned_data[column] >= limite_inferior) & (self.cleaned_data[column] <= limite_superior)]
        final_shape = self.cleaned_data.shape
        print(f"🧹 Outliers eliminados en '{column}': dataset pasó de {initial_shape[0]} a {final_shape[0]} filas.")

    def normalize_minmax(self, columns):
        for col in columns:
            if col not in self.cleaned_data.columns or not pd.api.types.is_numeric_dtype(self.cleaned_data[col]):
                print(f"Advertencia: Columna '{col}' no es numérica o no existe. No normalizada (Min-Max).")
                continue
            min_val = self.cleaned_data[col].min()
            max_val = self.cleaned_data[col].max()
            if max_val - min_val == 0:
                self.cleaned_data[col] = 0 
            else:
                self.cleaned_data[col] = (self.cleaned_data[col] - min_val) / (max_val - min_val)
        print(f"🔄 Normalización Min-Max aplicada (donde fue posible) a: {columns}")

    def standardize_zscore(self, columns):
        for col in columns:
            if col not in self.cleaned_data.columns or not pd.api.types.is_numeric_dtype(self.cleaned_data[col]):
                print(f"Advertencia: Columna '{col}' no es numérica o no existe. No estandarizada (Z-Score).")
                continue
            mean_val = self.cleaned_data[col].mean()
            std_val = self.cleaned_data[col].std()
            if std_val == 0:
                self.cleaned_data[col] = 0
            else:
                self.cleaned_data[col] = (self.cleaned_data[col] - mean_val) / std_val
        print(f"📈 Estandarización Z-Score aplicada (donde fue posible) a: {columns}")

    def one_hot_encode(self, columns):
        valid_cols_to_encode = [col for col in columns if col in self.cleaned_data.columns]
        if not valid_cols_to_encode:
            print("Advertencia: Ninguna de las columnas especificadas para One-Hot Encoding existe o es válida.")
            return
        
        print(f"Aplicando One-Hot Encoding a: {valid_cols_to_encode}")
        self.cleaned_data = pd.get_dummies(self.cleaned_data, columns=valid_cols_to_encode, drop_first=False)
        print(f"🏷️ One-Hot Encoding completado.")

    def save_clean_data(self, output_path):
        self.cleaned_data.to_csv(output_path, index=False)
        print(f"💾 Dataset limpio guardado exitosamente en: {output_path}")

    def rellenar_con_mediana(self, columna):
        if columna not in self.cleaned_data.columns:
            print(f"⚠️ La columna '{columna}' no existe en el DataFrame.")
            return
        if not pd.api.types.is_numeric_dtype(self.cleaned_data[columna]):
            print(f"⚠️ La columna '{columna}' no es numérica. No se puede calcular la mediana.")
            return
        if self.cleaned_data[columna].isnull().sum() == 0:
            print(f"ℹ️ No hay valores nulos en '{columna}' para rellenar.")
            return
            
        mediana = self.cleaned_data[columna].median()
        self.cleaned_data[columna].fillna(mediana, inplace=True)
        print(f"✔ Valores nulos en '{columna}' imputados con la mediana: {mediana}")

    def rellenar_con_moda(self, columna):
        if columna not in self.cleaned_data.columns:
            print(f"⚠️ La columna '{columna}' no existe en el DataFrame.")
            return
        if self.cleaned_data[columna].isnull().sum() == 0:
            print(f"ℹ️ No hay valores nulos en '{columna}' para rellenar.")
            return
        if self.cleaned_data[columna].isnull().all():
            print(f"⚠️ La columna '{columna}' está completamente vacía. No se puede calcular la moda.")
            return

        moda_series = self.cleaned_data[columna].mode()
        if not moda_series.empty:
            moda_val = moda_series[0]
            self.cleaned_data[columna].fillna(moda_val, inplace=True)
            print(f"✔ Valores nulos en '{columna}' imputados con la moda: {moda_val}")
        else:
             print(f"⚠️ No se pudo determinar la moda para la columna '{columna}'.")

    def eliminar_columna(self, columna):
        if columna in self.cleaned_data.columns:
            self.cleaned_data.drop(columns=[columna], inplace=True)
            print(f"❌ Columna '{columna}' eliminada correctamente.")
        else:
            print(f"⚠️ La columna '{columna}' no existe en el DataFrame.")

    def eliminar_filas_nulo(self, columna): # Solo en una columna específica
        if columna in self.cleaned_data.columns:
            filas_antes = self.cleaned_data.shape[0]
            self.cleaned_data.dropna(subset=[columna], inplace=True)
            filas_despues = self.cleaned_data.shape[0]
            eliminadas = filas_antes - filas_despues
            print(f"🧹 Se eliminaron {eliminadas} filas con nulos en la columna '{columna}'.")
        else:
            print(f"⚠️ La columna '{columna}' no existe en el DataFrame.")

    def detect_outliers_zscore(self, column, threshold=3):
        if column not in self.cleaned_data.columns:
            print(f"Advertencia: La columna '{column}' no existe para Z-score.")
            return pd.DataFrame()
        if not pd.api.types.is_numeric_dtype(self.cleaned_data[column]):
            print(f"Advertencia: La columna '{column}' no es numérica para Z-score.")
            return pd.DataFrame()
        if self.cleaned_data[column].std() == 0:
            print(f"⚠️ La desviación estándar de '{column}' es cero. No se pueden calcular Z-scores.")
            return pd.DataFrame()

        z_scores = np.abs((self.cleaned_data[column] - self.cleaned_data[column].mean()) / self.cleaned_data[column].std())
        outliers = self.cleaned_data[z_scores > threshold]
        print(f"🔎 Se detectaron {outliers.shape[0]} outliers en '{column}' (Z-score > {threshold}).")
        return outliers

    def normalizar_texto(self, texto_series): # Aplicar a una Series entera
        def _normalize(texto):
            if isinstance(texto, str):
                texto = texto.lower()
                texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
                texto = ' '.join(texto.split())
                return texto
            return texto
        return texto_series.apply(_normalize)

    def replace_unknown_error(self, column): 
        if column not in self.cleaned_data.columns:
            print(f"⚠️ La columna '{column}' no existe para reemplazar 'UNKNOWN' y 'ERROR'.")
            return
        self.cleaned_data[column] = self.cleaned_data[column].replace(['UNKNOWN', 'ERROR'], np.nan)
        print(f"✔ Se reemplazó 'UNKNOWN' y 'ERROR' en '{column}' por NaN (si existían).")