import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, dash_table
from statsmodels.tsa.stattools import acf

from estilos import (
    AZUL,
    AZUL_MED,
    BLANCO,
    CELESTE,
    FUENTE,
    TEXTO,
    estilo_flex,
    estilo_parrafo,
    estilo_parrafo_sec,
    estilo_tarjeta,
    estilo_titulo,
    tarjeta_metrica,
)


RUTA_METADATA_SVR = Path("Resultados/metadata_modelo_svr_calamar.json")
RUTA_TEST_SVR = Path("Resultados/test_final_externo_svr_calamar.csv")
RUTA_MODELO_SVR = Path("Resultados/modelo_svr_calamar.joblib")

RUTA_METADATA_KNN = Path("Resultados/metadata_modelo_knn_calamar.json")
RUTA_TEST_KNN = Path("Resultados/test_final_externo_knn_calamar.csv")
RUTA_MODELO_KNN = Path("Resultados/modelo_knn_calamar.joblib")

RUTA_METADATA_XARIMA = Path("Resultados/metadata_modelo_xarima_calamar.json")
RUTA_TEST_XARIMA = Path("Resultados/test_final_externo_xarima_calamar.csv")
RUTA_MODELO_XARIMA = Path("Resultados/modelo_xarima_calamar.joblib")

RUTA_METADATA_DT = Path("Resultados/metadata_modelo_dt_calamar.json")
RUTA_TEST_DT = Path("Resultados/test_final_externo_dt_calamar.csv")
RUTA_MODELO_DT = Path("Resultados/modelo_dt_calamar.joblib")

RUTA_METADATA_LASSO = Path("Resultados/metadata_modelo_lasso_calamar.json")
RUTA_TEST_LASSO = Path("Resultados/test_final_externo_lasso_calamar.csv")
RUTA_MODELO_LASSO = Path("Resultados/modelo_lasso_calamar.joblib")

RUTA_METADATA_RIDGE = Path("Resultados/metadata_modelo_ridge_calamar.json")
RUTA_TEST_RIDGE = Path("Resultados/test_final_externo_ridge_calamar.csv")
RUTA_MODELO_RIDGE = Path("Resultados/modelo_ridge_calamar.joblib")

RUTA_METADATA_RF = Path("Resultados/metadata_modelo_rf_calamar.json")
RUTA_TEST_RF = Path("Resultados/test_final_externo_rf_calamar.csv")
RUTA_MODELO_RF = Path("Resultados/modelo_rf_calamar.joblib")

RUTA_METADATA_XGB = Path("Resultados/metadata_modelo_xgb_calamar.json")
RUTA_TEST_XGB = Path("Resultados/test_final_externo_xgb_calamar.csv")
RUTA_MODELO_XGB = Path("Resultados/modelo_xgb_calamar.joblib")

RUTA_SERIE_COMPLETA = Path("data/Niveles_imputados_completo.csv")


def crear_tabla_simple(df, page_size=10):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in df.columns],
        page_size=page_size,
        style_table={"overflowX": "auto", "marginTop": "14px"},
        style_cell={
            "textAlign": "center",
            "fontFamily": FUENTE,
            "fontSize": "14px",
            "padding": "10px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#EAF1F8",
            "color": AZUL,
            "border": "1px solid #D9E2EF",
        },
        style_data={
            "backgroundColor": "white",
            "color": TEXTO,
            "border": "1px solid #D9E2EF",
        },
    )


def cargar_resultados_svr_calamar():
    with open(RUTA_METADATA_SVR, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_SVR, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_resultados_knn_calamar():
    with open(RUTA_METADATA_KNN, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_KNN, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test




def cargar_resultados_xarima_calamar():
    with open(RUTA_METADATA_XARIMA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_XARIMA, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_resultados_dt_calamar():
    with open(RUTA_METADATA_DT, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_DT, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_resultados_lasso_calamar():
    with open(RUTA_METADATA_LASSO, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_LASSO, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_resultados_ridge_calamar():
    with open(RUTA_METADATA_RIDGE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_RIDGE, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_resultados_rf_calamar():
    with open(RUTA_METADATA_RF, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_RF, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_resultados_xgb_calamar():
    with open(RUTA_METADATA_XGB, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_XGB, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def figura_serie_svr(df_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=2),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_predicho"],
        mode="lines",
        name="Calamar predicho - SVR",
        line=dict(color=CELESTE, width=2, dash="dash"),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel predicho:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=70, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig


def figura_serie_knn(df_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=2),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_predicho"],
        mode="lines",
        name="Calamar predicho - KNN",
        line=dict(color=CELESTE, width=2, dash="dash"),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel predicho:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=70, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig



def figura_serie_xarima(df_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=2),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_predicho"],
        mode="lines",
        name="Calamar predicho - XARIMA",
        line=dict(color=CELESTE, width=2, dash="dash"),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel predicho:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=70, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig




def figura_serie_dt(df_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=2),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_predicho"],
        mode="lines",
        name="Calamar predicho - Decision Tree",
        line=dict(color=CELESTE, width=2, dash="dash"),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel predicho:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=70, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig



def figura_serie_lasso(df_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=2),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_predicho"],
        mode="lines",
        name="Calamar predicho - Lasso",
        line=dict(color=CELESTE, width=2, dash="dash"),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel predicho:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=70, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig


def figura_serie_ridge(df_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=2),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_predicho"],
        mode="lines",
        name="Calamar predicho - Ridge",
        line=dict(color=CELESTE, width=2, dash="dash"),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel predicho:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=70, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig


def figura_serie_rf(df_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=2),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_predicho"],
        mode="lines",
        name="Calamar predicho - Random Forest",
        line=dict(color=CELESTE, width=2, dash="dash"),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel predicho:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=70, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig


def figura_serie_xgb(df_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=2),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_test["Fecha"],
        y=df_test["Calamar_predicho"],
        mode="lines",
        name="Calamar predicho - XGBoost",
        line=dict(color=CELESTE, width=2, dash="dash"),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel predicho:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=70, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig


def leer_serie_completa_calamar():
    df_serie = pd.read_csv(
        RUTA_SERIE_COMPLETA,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )

    df_serie.columns = df_serie.columns.str.strip()

    for col in df_serie.columns:
        if "Fecha" in col:
            df_serie = df_serie.rename(columns={col: "Fecha"})

    df_serie["Fecha"] = pd.to_datetime(df_serie["Fecha"], errors="coerce")

    if "Calamar" not in df_serie.columns:
        raise ValueError("No se encontró la columna Calamar en la serie completa.")

    df_serie["Calamar"] = (
        df_serie["Calamar"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df_serie["Calamar"] = pd.to_numeric(df_serie["Calamar"], errors="coerce")

    df_serie = df_serie[["Fecha", "Calamar"]].dropna().sort_values("Fecha")

    return df_serie


def figura_particion_temporal_svr(metadata, df_serie, df_test):
    fecha_inicio_trainval = pd.to_datetime(metadata["fecha_inicio_trainval"])
    fecha_fin_trainval = pd.to_datetime(metadata["fecha_fin_trainval"])
    fecha_inicio_test = pd.to_datetime(metadata["fecha_inicio_test_externo"])
    fecha_fin_test = pd.to_datetime(metadata["fecha_fin_test_externo"])

    df_serie = df_serie.copy()
    df_test = df_test.copy()

    df_serie["Fecha"] = pd.to_datetime(df_serie["Fecha"], errors="coerce")
    df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    df_trainval = df_serie[
        (df_serie["Fecha"] >= fecha_inicio_trainval)
        & (df_serie["Fecha"] <= fecha_fin_trainval)
    ].copy()

    df_test_obs = df_serie[
        (df_serie["Fecha"] >= fecha_inicio_test)
        & (df_serie["Fecha"] <= fecha_fin_test)
    ].copy()

    fig = go.Figure()

    # Serie completa como referencia
    fig.add_trace(
        go.Scatter(
            x=df_serie["Fecha"],
            y=df_serie["Calamar"],
            mode="lines",
            name="Serie completa",
            line=dict(
                color="rgba(30,30,30,0.45)",
                width=1.4,
                dash="dash"
            ),
            hovertemplate=(
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )
        )
    )

    # Train / validación interna
    fig.add_trace(
        go.Scatter(
            x=df_trainval["Fecha"],
            y=df_trainval["Calamar"],
            mode="lines",
            name="Train / validación interna",
            line=dict(
                color="#D94F8C",
                width=2.3
            ),
            hovertemplate=(
                "<b>Train/validación interna</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )
        )
    )

    # Test observado
    fig.add_trace(
        go.Scatter(
            x=df_test_obs["Fecha"],
            y=df_test_obs["Calamar"],
            mode="lines",
            name="Test observado",
            line=dict(
                color="#3498DB",
                width=2.8
            ),
            hovertemplate=(
                "<b>Test observado</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )
        )
    )

    # Predicción SVR en test externo
    fig.add_trace(
        go.Scatter(
            x=df_test["Fecha"],
            y=df_test["Calamar_predicho"],
            mode="lines",
            name="Predicción SVR",
            line=dict(
                color="#1A3A5C",
                width=2.4,
                dash="dot"
            ),
            hovertemplate=(
                "<b>Predicción SVR</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )
        )
    )

    # Líneas verticales de separación
    fig.add_vline(
        x=fecha_fin_trainval,
        line_width=1.8,
        line_dash="dash",
        line_color="#8E44AD"
    )

    fig.add_vline(
        x=fecha_inicio_test,
        line_width=1.8,
        line_dash="dash",
        line_color="#3498DB"
    )

    y_min = df_serie["Calamar"].min()
    y_max = df_serie["Calamar"].max()
    rango_y = y_max - y_min
    margen_y = 0.12 * rango_y

    # Etiquetas superiores
    if len(df_trainval) > 0:
        fig.add_annotation(
            x=df_trainval["Fecha"].iloc[len(df_trainval) // 2],
            y=y_max + margen_y,
            text="<b>Train / validación interna</b>",
            showarrow=False,
            font=dict(
                family="Georgia",
                size=14,
                color="#D94F8C"
            )
        )

    if len(df_test_obs) > 0:
        fig.add_annotation(
            x=df_test_obs["Fecha"].iloc[len(df_test_obs) // 2],
            y=y_max + margen_y,
            text="<b>Test externo</b>",
            showarrow=False,
            font=dict(
                family="Georgia",
                size=14,
                color="#3498DB"
            )
        )

    fig.add_annotation(
        x=fecha_inicio_test,
        y=y_max,
        text="observado<br>SVR",
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(
            family="Georgia",
            size=12,
            color="#1A3A5C"
        ),
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor="rgba(26,58,92,0.15)",
        borderwidth=1
    )

    fig.update_layout(
        title=None,
        xaxis_title="Fecha",
        yaxis_title="Nivel en Calamar [cm]",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Georgia",
            size=13,
            color="#1A3A5C"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(26,58,92,0.18)",
            borderwidth=1
        ),
        margin=dict(l=70, r=40, t=90, b=60),
        height=520
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#D9E2EF"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#D9E2EF",
        zeroline=False,
        range=[
            max(0, y_min - 0.04 * rango_y),
            y_max + 1.45 * margen_y
        ]
    )

    return fig


def figura_particion_temporal_knn(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para KNN sin duplicar toda la lógica de partición temporal
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción KNN"
            trace.hovertemplate = (
                "<b>Predicción KNN</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>KNN"

    return fig




def figura_particion_temporal_xarima(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para XARIMA sin duplicar toda la lógica de partición temporal
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción XARIMA"
            trace.hovertemplate = (
                "<b>Predicción XARIMA</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>XARIMA"

    return fig


def figura_particion_temporal_dt(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para Decision Tree sin duplicar toda la lógica
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción Decision Tree"
            trace.hovertemplate = (
                "<b>Predicción Decision Tree</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>Decision Tree"

    return fig


def figura_particion_temporal_lasso(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para Lasso sin duplicar toda la lógica
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción Lasso"
            trace.hovertemplate = (
                "<b>Predicción Lasso</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>Lasso"

    return fig


def figura_particion_temporal_ridge(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para Ridge sin duplicar toda la lógica
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción Ridge"
            trace.hovertemplate = (
                "<b>Predicción Ridge</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>Ridge"

    return fig


def figura_particion_temporal_rf(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para Random Forest sin duplicar toda la lógica de partición temporal
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción Random Forest"
            trace.hovertemplate = (
                "<b>Predicción Random Forest</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>RF"

    return fig


def figura_particion_temporal_xgb(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para XGBoost sin duplicar toda la lógica
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción XGBoost"
            trace.hovertemplate = (
                "<b>Predicción XGBoost</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>XGBoost"

    return fig


def figura_histograma_residuos(df_test):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_test["Residuo"].dropna(),
        nbinsx=30,
        name="Residuos",
        marker_color=AZUL_MED,
        opacity=0.78,
        hovertemplate="<b>Residuo:</b> %{x:.2f}<br><b>Frecuencia:</b> %{y}<br><extra></extra>",
    ))

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=2,
        line_color="#B23A48",
        annotation_text="Residuo = 0",
        annotation_position="top right",
    )

    fig.update_layout(
        title=None,
        xaxis_title="Residuo [cm]",
        yaxis_title="Frecuencia",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        bargap=0.05,
        margin=dict(l=70, r=40, t=50, b=60),
        height=440,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig


def figura_acf_residuos(df_test, nlags=60):
    residuos = df_test["Residuo"].dropna().values
    acf_vals = acf(residuos, nlags=nlags, fft=True)
    lags = np.arange(len(acf_vals))
    conf = 1.96 / np.sqrt(len(residuos))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lags,
        y=acf_vals,
        name="ACF residuos",
        marker_color=AZUL_MED,
        opacity=0.9,
        hovertemplate="<b>Rezago:</b> %{x}<br><b>ACF:</b> %{y:.3f}<br><extra></extra>",
    ))
    fig.add_hline(y=conf, line_dash="dash", line_width=1, line_color="#B23A48")
    fig.add_hline(y=-conf, line_dash="dash", line_width=1, line_color="#B23A48")
    fig.add_hline(y=0, line_width=1, line_color=AZUL)

    fig.update_layout(
        title=None,
        xaxis_title="Rezago",
        yaxis_title="Autocorrelación",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=70, r=40, t=50, b=60),
        height=440,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig


def layout_svr_calamar():
    metadata, df_test = cargar_resultados_svr_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_svr(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_svr(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)



    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "C", "Valor": best_params["svr__C"]},
        {"Parámetro": "epsilon", "Valor": best_params["svr__epsilon"]},
        {"Parámetro": "gamma", "Valor": best_params["svr__gamma"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Máquina de Vectores de Soporte (SVR) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y SVR, aplicado a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
                style=estilo_parrafo,
            ),
            html.P(metadata["criterio_final"], style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P("El último año disponible se reservó como test externo final. El resto de la serie se empleó para entrenamiento y validación interna.", style=estilo_parrafo),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros del SVR. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
            crear_tabla_simple(df_hiper, page_size=10),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[

    html.H2(
        "Partición temporal del modelado",
        style=estilo_titulo
    ),

    html.P(
        "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, "
        "seguido por un test externo final. El último año fue reservado como conjunto externo "
        "para evaluar el desempeño del modelo sobre datos no utilizados durante la selección "
        "de hiperparámetros.",
        style=estilo_parrafo_sec
    ),

    dcc.Graph(
        figure=fig_particion,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "particion_temporal_svr_calamar",
                "height": 900,
                "width": 1400,
                "scale": 2
            }
        }
    )

]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo SVR durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])



def layout_knn_calamar():
    metadata, df_test = cargar_resultados_knn_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_knn(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_knn(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "n_neighbors", "Valor": best_params["knn__n_neighbors"]},
        {"Parámetro": "weights", "Valor": best_params["knn__weights"]},
        {"Parámetro": "p", "Valor": best_params["knn__p"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("K-Vecinos más Cercanos (KNN) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y KNeighborsRegressor, aplicado a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
                style=estilo_parrafo,
            ),
            html.P(metadata["criterio_final"], style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P("El último año disponible se reservó como test externo final. El resto de la serie se empleó para entrenamiento y validación interna.", style=estilo_parrafo),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros del KNN. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
            crear_tabla_simple(df_hiper, page_size=10),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, "
                "seguido por un test externo final. El último año fue reservado como conjunto externo "
                "para evaluar el desempeño del modelo sobre datos no utilizados durante la selección "
                "de hiperparámetros.",
                style=estilo_parrafo_sec
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_knn_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo KNN durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])




def layout_xarima_calamar():
    metadata, df_test = cargar_resultados_xarima_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_xarima(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_xarima(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    order = metadata["order"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['train_size']} días"},
        {"Parámetro": "Orden ARIMA (p, d, q)", "Valor": f"({order[0]}, {order[1]}, {order[2]})"},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Órdenes evaluados", "Valor": "; ".join([f"({o[0]}, {o[1]}, {o[2]})" for o in metadata["ordenes_evaluados"]])},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("XARIMA/ARIMA con SARIMAX - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una formulación XARIMA/ARIMA implementada con SARIMAX, aplicada a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
                style=estilo_parrafo,
            ),
            html.P(metadata["criterio_final"], style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P("El último año disponible se reservó como test externo final. El resto de la serie se empleó para entrenamiento y validación interna.", style=estilo_parrafo),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P("La búsqueda evaluó diferentes ventanas de entrenamiento y órdenes ARIMA. La tabla resume la configuración seleccionada y las alternativas evaluadas.", style=estilo_parrafo),
            crear_tabla_simple(df_hiper, page_size=10),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, "
                "seguido por un test externo final. El último año fue reservado como conjunto externo "
                "para evaluar el desempeño del modelo sobre datos no utilizados durante la selección "
                "de hiperparámetros.",
                style=estilo_parrafo_sec
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_xarima_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo XARIMA durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


def layout_dt_calamar():
    metadata, df_test = cargar_resultados_dt_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_dt(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_dt(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    def _valor_parametro(valor):
        return "None" if valor is None else valor

    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "max_depth", "Valor": _valor_parametro(best_params["dt__max_depth"])},
        {"Parámetro": "min_samples_split", "Valor": best_params["dt__min_samples_split"]},
        {"Parámetro": "min_samples_leaf", "Valor": best_params["dt__min_samples_leaf"]},
        {"Parámetro": "max_features", "Valor": _valor_parametro(best_params["dt__max_features"])},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Árbol de Decisión - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un DecisionTreeRegressor, aplicado a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
                style=estilo_parrafo,
            ),
            html.P(metadata["criterio_final"], style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P("El último año disponible se reservó como test externo final. El resto de la serie se empleó para entrenamiento y validación interna.", style=estilo_parrafo),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros del árbol de decisión. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
            crear_tabla_simple(df_hiper, page_size=10),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, "
                "seguido por un test externo final. El último año fue reservado como conjunto externo "
                "para evaluar el desempeño del modelo sobre datos no utilizados durante la selección "
                "de hiperparámetros.",
                style=estilo_parrafo_sec
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_dt_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo de Árbol de Decisión durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])




def layout_lasso_calamar():
    metadata, df_test = cargar_resultados_lasso_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_lasso(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_lasso(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "alpha", "Valor": best_params["lasso__alpha"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Regresión Lasso - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y Lasso, aplicado a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
                style=estilo_parrafo,
            ),
            html.P(metadata["criterio_final"], style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P("El último año disponible se reservó como test externo final. El resto de la serie se empleó para entrenamiento y validación interna.", style=estilo_parrafo),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P("La búsqueda evaluó distintas ventanas de entrada y valores del parámetro alpha de Lasso. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
            crear_tabla_simple(df_hiper, page_size=10),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, "
                "seguido por un test externo final. El último año fue reservado como conjunto externo "
                "para evaluar el desempeño del modelo sobre datos no utilizados durante la selección "
                "de hiperparámetros.",
                style=estilo_parrafo_sec
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_lasso_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo Lasso durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


def layout_ridge_calamar():
    metadata, df_test = cargar_resultados_ridge_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_ridge(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_ridge(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "alpha", "Valor": best_params["ridge__alpha"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Regresión Ridge - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y Ridge, aplicado a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
                style=estilo_parrafo,
            ),
            html.P(metadata["criterio_final"], style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P("El último año disponible se reservó como test externo final. El resto de la serie se empleó para entrenamiento y validación interna.", style=estilo_parrafo),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P("La búsqueda evaluó distintas ventanas de entrada y valores del parámetro alpha de Ridge. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
            crear_tabla_simple(df_hiper, page_size=10),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, "
                "seguido por un test externo final. El último año fue reservado como conjunto externo "
                "para evaluar el desempeño del modelo sobre datos no utilizados durante la selección "
                "de hiperparámetros.",
                style=estilo_parrafo_sec
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_ridge_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo Ridge durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


def layout_rf_calamar():
    metadata, df_test = cargar_resultados_rf_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_rf(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_rf(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "n_estimators", "Valor": best_params["rf__n_estimators"]},
        {"Parámetro": "max_depth", "Valor": best_params["rf__max_depth"]},
        {"Parámetro": "min_samples_leaf", "Valor": best_params["rf__min_samples_leaf"]},
        {"Parámetro": "max_features", "Valor": best_params["rf__max_features"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Random Forest - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por RandomForestRegressor, aplicado a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
                style=estilo_parrafo,
            ),
            html.P(metadata["criterio_final"], style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P("El último año disponible se reservó como test externo final. El resto de la serie se empleó para entrenamiento y validación interna.", style=estilo_parrafo),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros del Random Forest. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
            crear_tabla_simple(df_hiper, page_size=10),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, "
                "seguido por un test externo final. El último año fue reservado como conjunto externo "
                "para evaluar el desempeño del modelo sobre datos no utilizados durante la selección "
                "de hiperparámetros.",
                style=estilo_parrafo_sec
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_rf_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo Random Forest durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


def layout_xgb_calamar():
    metadata, df_test = cargar_resultados_xgb_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_xgb(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_xgb(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "n_estimators", "Valor": best_params["xgb__n_estimators"]},
        {"Parámetro": "max_depth", "Valor": best_params["xgb__max_depth"]},
        {"Parámetro": "learning_rate", "Valor": best_params["xgb__learning_rate"]},
        {"Parámetro": "subsample", "Valor": best_params["xgb__subsample"]},
        {"Parámetro": "colsample_bytree", "Valor": best_params["xgb__colsample_bytree"]},
        {"Parámetro": "reg_lambda", "Valor": best_params["xgb__reg_lambda"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("XGBoost - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por XGBRegressor, aplicado a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
                style=estilo_parrafo,
            ),
            html.P(metadata["criterio_final"], style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P("El último año disponible se reservó como test externo final. El resto de la serie se empleó para entrenamiento y validación interna.", style=estilo_parrafo),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros del XGBoost. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
            crear_tabla_simple(df_hiper, page_size=10),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, "
                "seguido por un test externo final. El último año fue reservado como conjunto externo "
                "para evaluar el desempeño del modelo sobre datos no utilizados durante la selección "
                "de hiperparámetros.",
                style=estilo_parrafo_sec
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_xgb_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo XGBoost durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


def layout_modelos(modelos=None):
    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Modelos implementados", style=estilo_titulo),
            html.P(
                "Seleccione un modelo para visualizar su configuración, validación temporal, métricas, predicciones y diagnóstico de residuos.",
                style=estilo_parrafo_sec,
            ),
            dcc.Dropdown(
                id="selector-modelo",
                options=[
                    {"label": "Máquina de Vectores de Soporte (SVR)", "value": "svr_calamar"},
                    {"label": "K-Vecinos más Cercanos (KNN)", "value": "knn_calamar"},
                    {"label": "XARIMA/ARIMA", "value": "xarima_calamar"},
                    {"label": "Árbol de Decisión", "value": "dt_calamar"},
                    {"label": "Regresión Lasso", "value": "lasso_calamar"},
                    {"label": "Regresión Ridge", "value": "ridge_calamar"},
                    {"label": "Random Forest", "value": "rf_calamar"},
                    {"label": "XGBoost", "value": "xgb_calamar"},
                ],
                value="svr_calamar",
                clearable=False,
                style={"fontFamily": FUENTE, "fontSize": "14px", "maxWidth": "420px"},
            ),
        ]),
        html.Div(id="contenido-modelo"),
    ])


def registrar_callbacks_modelos(app, df, serie_objetivo, modelos=None):
    @app.callback(Output("contenido-modelo", "children"), Input("selector-modelo", "value"))
    def mostrar_modelo(modelo):
        if modelo == "svr_calamar":
            return layout_svr_calamar()

        if modelo == "knn_calamar":
            return layout_knn_calamar()

        if modelo == "xarima_calamar":
            return layout_xarima_calamar()

        if modelo == "dt_calamar":
            return layout_dt_calamar()

        if modelo == "lasso_calamar":
            return layout_lasso_calamar()

        if modelo == "ridge_calamar":
            return layout_ridge_calamar()

        if modelo == "rf_calamar":
            return layout_rf_calamar()

        if modelo == "xgb_calamar":
            return layout_xgb_calamar()

        return html.Div(style=estilo_tarjeta, children=[
            html.H2("Modelo no disponible", style=estilo_titulo),
            html.P("La información de este modelo todavía no ha sido cargada.", style=estilo_parrafo),
        ])
