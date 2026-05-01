from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data") / "Niveles_alineados.csv"

NOMBRES_ESTACIONES = {
    "Achi": "Achi",
    "Calamar": "Calamar",
    "ElBanco": "El Banco",
    "SaladoBlanco": "Salado Blanco",
    "PuertoBerrio": "Puerto Berrío",
    "Barrancabermeja": "Barrancabermeja",
}

COLORES_ESTACIONES = {
    "Achi": "#1f77b4",
    "Calamar": "#d62728",
    "ElBanco": "#2ca02c",
    "SaladoBlanco": "#9467bd",
    "PuertoBerrio": "#ff7f0e",
    "Barrancabermeja": "#19d3f3",
}


def cargar_datos():
    """Carga los datos reales. Si no encuentra el CSV, crea datos de respaldo."""

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    else:
        np.random.seed(42)
        fechas = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        df = pd.DataFrame({"Fecha": fechas})
        base = 500 + np.cumsum(np.random.randn(len(fechas)) * 4)
        df["Achi"] = base + np.random.randn(len(fechas)) * 20
        df["Calamar"] = base * 0.8 + np.random.randn(len(fechas)) * 18
        df["ElBanco"] = base * 0.9 + np.random.randn(len(fechas)) * 16
        df["SaladoBlanco"] = base * 0.7 + np.random.randn(len(fechas)) * 15
        df["PuertoBerrio"] = base * 0.6 + np.random.randn(len(fechas)) * 14
        df["Barrancabermeja"] = base * 0.65 + np.random.randn(len(fechas)) * 14

    columnas_estaciones = [col for col in NOMBRES_ESTACIONES if col in df.columns]
    serie_objetivo = "Calamar" if "Calamar" in df.columns else columnas_estaciones[0]

    return df, columnas_estaciones, serie_objetivo
