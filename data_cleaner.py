import numpy as np
import os
import unicodedata
import pandas as pd

class DataCleaner:
    """
    Clase para realizar diversas operaciones de limpieza y preprocesamiento de datos
    en un DataFrame de Pandas.

    Atributos:
        data (pd.DataFrame): El DataFrame original cargado.
        cleaned_data (pd.DataFrame): Una copia del DataFrame original que se modifica
                                     a medida que se aplican los métodos de limpieza.
    """
    def __init__(self, filepath_or_buffer):
        """
        Inicializa el objeto DataCleaner cargando datos desde un archivo CSV o un buffer.

        FileNotFoundError: Si `filepath_or_buffer` es una cadena de ruta y el archivo no se encuentra.
        Exception: Si ocurre un error al leer el archivo CSV.
        """
        if isinstance(filepath_or_buffer, str): # Si es una ruta de archivo
            if not os.path.exists(filepath_or_buffer):
                raise FileNotFoundError(f"No se encontró el archivo: {filepath_or_buffer}")
            self.data = pd.read_csv(filepath_or_buffer)
        else: 
            self.data = pd.read_csv(filepath_or_buffer)

        self.cleaned_data = self.data.copy()
       
        print(f"Dataset cargado exitosamente con {self.cleaned_data.shape[0]} filas y {self.cleaned_data.shape[1]} columnas.")

    def report_nulls(self) -> pd.Series:
        """
        Genera e imprime un reporte de la cantidad de valores nulos por columna
        en el DataFrame `cleaned_data`.

        pd.Series: Una serie de Pandas donde el índice son los nombres de las columnas
                       y los valores son la cantidad de nulos en cada una. Solo se incluyen
                       columnas con al menos un valor nulo.
        """
        null_report = self.cleaned_data.isnull().sum()
        print("Reporte de valores nulos por columna:")
        print(null_report[null_report > 0])
        return null_report

    def detect_outliers_iqr(self, column: str) -> pd.DataFrame:
        """
        Detecta outliers en una columna numérica especificada utilizando el método del Rango Intercuartílico (IQR).

        Un valor se considera outlier si está por debajo de Q1 - 1.5*IQR o por encima de Q3 + 1.5*IQR.


        column (str): El nombre de la columna en `cleaned_data` donde se detectarán los outliers.

      
        pd.DataFrame: Un DataFrame que contiene las filas identificadas como outliers.
                          Retorna un DataFrame vacío si la columna no existe, no es numérica,
                          o si el IQR es cero.
        """
        if column not in self.cleaned_data.columns:
            print(f"Advertencia: La columna '{column}' no existe para detectar outliers.")
            return pd.DataFrame()
        if not pd.api.types.is_numeric_dtype(self.cleaned_data[column]):
            print(f"Advertencia: La columna '{column}' no es numérica. No se pueden detectar outliers con IQR.")
            return pd.DataFrame()

        Q1 = self.cleaned_data[column].quantile(0.25)
        Q3 = self.cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0: # Evitar problemas si todos los valores son iguales
            print(f"Rango intercuartílico (IQR) es cero para la columna '{column}'. No se detectarán outliers por este método aquí.")
            return pd.DataFrame()

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = self.cleaned_data[(self.cleaned_data[column] < limite_inferior) | (self.cleaned_data[column] > limite_superior)]
        print(f"🔎 Se detectaron {outliers.shape[0]} outliers en la columna '{column}' usando IQR.")
        return outliers

    def remove_outliers_iqr(self, column: str) -> None:
        """
        Elimina outliers de una columna numérica especificada utilizando el método del Rango Intercuartílico (IQR).
        Las filas que contienen outliers en la columna especificada son eliminadas de `cleaned_data`.

        column (str): El nombre de la columna en `cleaned_data` de la cual se eliminarán los outliers.
                          No hace nada si la columna no existe, no es numérica, o si el IQR es cero.
        """
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

    def normalize_minmax(self, columns: list[str]) -> None:
        """
        Aplica la normalización Min-Max a las columnas numéricas especificadas en `cleaned_data`.
        La normalización escala los datos al rango [0, 1].
        Si una columna tiene todos sus valores iguales (rango max-min es 0), sus valores se establecen a 0.

        columns (list[str]): Una lista de nombres de columnas a normalizar.
                                 Las columnas no numéricas o no existentes serán ignoradas con una advertencia.
        """
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

    def standardize_zscore(self, columns: list[str]) -> None:
        """
        Aplica la estandarización Z-score (media 0, desviación estándar 1) a las columnas
        numéricas especificadas en `cleaned_data`.
        Si una columna tiene una desviación estándar de 0, sus valores se establecen a 0.

        """
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

    def one_hot_encode(self, columns: list[str]) -> None:
        """
        Aplica One-Hot Encoding a las columnas categóricas especificadas en `cleaned_data`.
        Esto crea nuevas columnas binarias para cada categoría única en las columnas originales.

        """
        valid_cols_to_encode = [col for col in columns if col in self.cleaned_data.columns]
        if not valid_cols_to_encode:
            print("Advertencia: Ninguna de las columnas especificadas para One-Hot Encoding existe o es válida.")
            return

        print(f"Aplicando One-Hot Encoding a: {valid_cols_to_encode}")
        self.cleaned_data = pd.get_dummies(self.cleaned_data, columns=valid_cols_to_encode, drop_first=False)
        print(f"🏷️ One-Hot Encoding completado.")

    def save_clean_data(self, output_path: str) -> None:
        """
        Guarda el DataFrame `cleaned_data` en un archivo CSV.

        """
        self.cleaned_data.to_csv(output_path, index=False)
        print(f"💾 Dataset limpio guardado exitosamente en: {output_path}")

    def rellenar_con_mediana(self, columna: str) -> None:
        """
        Rellena los valores nulos (NaN) en una columna numérica especificada de `cleaned_data`
        con la mediana de esa columna.

        columna (str): El nombre de la columna a imputar.
                           No hace nada si la columna no existe, no es numérica, o no tiene nulos.
        """
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

    def rellenar_con_moda(self, columna: str) -> None:
        """
        Rellena los valores nulos (NaN) en una columna especificada de `cleaned_data`
        con la moda (valor más frecuente) de esa columna. Si hay múltiples modas,
        se utiliza la primera.

        columna (str): El nombre de la columna a imputar.
                           No hace nada si la columna no existe, no tiene nulos, está completamente vacía,
                           o no se puede determinar la moda.
        """
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

    def eliminar_columna(self, columna: str) -> None:
        """
        Elimina una columna especificada del DataFrame `cleaned_data`.

        columna (str): El nombre de la columna a eliminar.
                           No hace nada si la columna no existe.
        """
        if columna in self.cleaned_data.columns:
            self.cleaned_data.drop(columns=[columna], inplace=True)
            print(f"❌ Columna '{columna}' eliminada correctamente.")
        else:
            print(f"⚠️ La columna '{columna}' no existe en el DataFrame.")

    def eliminar_filas_nulo(self, columna: str) -> None:
        """
        Elimina las filas que contienen valores nulos (NaN) en una columna específica
        del DataFrame `cleaned_data`.

     
        columna (str): El nombre de la columna a verificar para valores nulos.
                           Las filas con NaN en esta columna serán eliminadas.
                           No hace nada si la columna no existe.
        """
        if columna in self.cleaned_data.columns:
            filas_antes = self.cleaned_data.shape[0]
            self.cleaned_data.dropna(subset=[columna], inplace=True)
            filas_despues = self.cleaned_data.shape[0]
            eliminadas = filas_antes - filas_despues
            print(f"🧹 Se eliminaron {eliminadas} filas con nulos en la columna '{columna}'.")
        else:
            print(f"⚠️ La columna '{columna}' no existe en el DataFrame.")

    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detecta outliers en una columna numérica utilizando el método Z-score.
        Un valor se considera outlier si su Z-score absoluto es mayor que el umbral especificado.

        column (str): El nombre de la columna numérica donde se detectarán los outliers.
            threshold (float, optional): El umbral de Z-score para considerar un valor como outlier.
                                         Por defecto es 3.0.

        pd.DataFrame: Un DataFrame que contiene las filas identificadas como outliers.
                          Retorna un DataFrame vacío si la columna no existe, no es numérica,
                          o si su desviación estándar es cero.
        """
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

    def normalizar_texto(self, texto_series: pd.Series) -> pd.Series:
        """
        Normaliza una serie de Pandas que contiene texto.
        Los pasos de normalización incluyen:
        1. Conversión a minúsculas.
        2. Descomposición de caracteres Unicode (NFKD).
        3. Codificación a ASCII ignorando caracteres no representables.
        4. Decodificación a UTF-8.
        5. Eliminación de espacios en blanco redundantes.


        texto_series (pd.Series): La serie de Pandas que contiene los textos a normalizar.

       
        pd.Series: Una nueva serie con los textos normalizados.
                       Si un elemento de la serie no es una cadena, se devuelve tal cual.
        """
        def _normalize(texto: str | any) -> str | any:
            """Función auxiliar para normalizar un único texto."""
            if isinstance(texto, str):
                texto = texto.lower()
                texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
                texto = ' '.join(texto.split()) # Elimina espacios extra y saltos de línea
                return texto
            return texto # Devuelve el valor original si no es string (ej. NaN)
        return texto_series.apply(_normalize)

    def replace_unknown_error(self, column: str) -> None:
        """
        Reemplaza los valores 'UNKNOWN' y 'ERROR' (insensible a mayúsculas/minúsculas si
        se normaliza previamente el texto) con NaN en la columna especificada de `cleaned_data`.
        Este método asume que los valores son exactamente 'UNKNOWN' o 'ERROR'.
        Para una coincidencia insensible a mayúsculas/minúsculas, se debería aplicar `normalizar_texto`
        a la columna antes de usar este método.


        column (str): El nombre de la columna donde se reemplazarán los valores.
                           No hace nada si la columna no existe.
        """
        if column not in self.cleaned_data.columns:
            print(f"⚠️ La columna '{column}' no existe para reemplazar 'UNKNOWN' y 'ERROR'.")
            return
        # Se reemplazan directamente. Para manejo de may/min, normalizar antes.
        self.cleaned_data[column] = self.cleaned_data[column].replace(['UNKNOWN', 'ERROR'], np.nan)
        print(f"✔ Se reemplazó 'UNKNOWN' y 'ERROR' en '{column}' por NaN (si existían).")