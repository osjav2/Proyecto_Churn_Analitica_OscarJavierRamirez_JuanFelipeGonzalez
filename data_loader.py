# data_loader.py
import pandas as pd
from pathlib import Path

# --- VARIABLE EXPORTADA PARA EL NOTEBOOK ---
TARGET_COL = 'attrition_flag'

# ------------------------------------------

def load_and_map_data(file_path):
    """
    Carga los datos desde el archivo Excel, limpia los nombres de columnas
    y mapea la variable objetivo 'attrition_flag' a 0 (Existing) y 1 (Attrited).
    """
    print("Cargando y preparando los datos...")
    path = Path(file_path)  # Usa Path para manejo de rutas
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo en la ruta: {file_path}")
        return None
    except Exception as e:
        print(f"ERROR al cargar el archivo: {e}")
        return None

    # 1. Limpieza de nombres de columnas (minúsculas y sin espacios)
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False)

    # 2. Mapeo de la Variable Objetivo (Churn: 1, No Churn: 0)
    mapping = {
        'attrited customer': 1,
        'existing customer': 0
    }
    # Aseguramos que la columna esté en minúsculas para el mapeo
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower().map(mapping)

    # Manejar posibles valores NaT (Not a Time) o NaN si el mapeo falla en algún caso
    # Remplazamos valores no mapeados con la moda o se elimina (aquí se asume que todo mapea bien)

    print("Datos cargados, columnas limpiadas y target mapeado a 0/1 exitosamente.")
    return df