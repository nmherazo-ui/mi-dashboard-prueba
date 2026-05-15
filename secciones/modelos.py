import json
from pathlib import Path
from plotly.subplots import make_subplots
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

RUTA_METADATA_MLP = Path("Resultados/metadata_modelo_mlp_calamar.json")
RUTA_TEST_MLP = Path("Resultados/test_final_externo_mlp_calamar.csv")
RUTA_MODELO_MLP = Path("Resultados/modelo_mlp_calamar.joblib")

RUTA_METADATA_RNN = Path("Resultados/metadata_modelo_rnn_calamar.json")
RUTA_TEST_RNN = Path("Resultados/test_final_externo_rnn_calamar.csv")
RUTA_MODELO_RNN = Path("Resultados/modelo_rnn_calamar.keras")

RUTA_METADATA_LSTM = Path("Resultados/metadata_modelo_lstm_calamar.json")
RUTA_TEST_LSTM = Path("Resultados/test_final_externo_lstm_calamar.csv")
RUTA_MODELO_LSTM = Path("Resultados/modelo_lstm_calamar.keras")

RUTA_METADATA_CNN = Path("Resultados/metadata_modelo_cnn_calamar.json")
RUTA_TEST_CNN = Path("Resultados/test_final_externo_cnn_calamar.csv")
RUTA_MODELO_CNN = Path("Resultados/modelo_cnn_calamar.keras")

RUTA_SERIE_COMPLETA = Path("data/Niveles_imputados_completo.csv")

RUTA_SERIE_COMPLETA = Path("data/Niveles_imputados_completo.csv")

RUTAS_DM_ABSOLUTA = [
    Path("Resultados/matriz_pvalues_dm_absoluta.csv"),
    Path("Resultados/matriz_pvalues_dm_absoluta.xlsx"),
]

RUTAS_DM_CUADRATICA = [
    Path("Resultados/matriz_pvalues_dm_cuadratica.csv"),
    Path("Resultados/matriz_pvalues_dm_cuadratica.xlsx"),
]

RUTAS_MW_ERRORES_ABSOLUTOS = [
    Path("Resultados/matriz_errores_absolutos_modelos.csv"),
    Path("Resultados/matriz_errores_absolutos_modelos.xlsx"),
]

RUTAS_MW_SVR_VS_MODELOS = [
    Path("Resultados/resultados_mannwhitney_svr_vs_modelos.csv"),
    Path("Resultados/resultados_mannwhitney_svr_vs_modelos.xlsx"),
]

RUTAS_MW_TODOS_CONTRA_TODOS = [
    Path("Resultados/resultados_mannwhitney_todos_contra_todos.csv"),
    Path("Resultados/resultados_mannwhitney_todos_contra_todos.xlsx"),
]


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


def cargar_resultados_mlp_calamar():
    with open(RUTA_METADATA_MLP, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_MLP, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test





def cargar_resultados_rnn_calamar():
    with open(RUTA_METADATA_RNN, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_RNN, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test



def cargar_resultados_lstm_calamar():
    with open(RUTA_METADATA_LSTM, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_LSTM, sep=None, engine="python", encoding="utf-8-sig")
    df_test.columns = df_test.columns.str.strip()

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_resultados_cnn_calamar():
    with open(RUTA_METADATA_CNN, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = pd.read_csv(RUTA_TEST_CNN, sep=None, engine="python", encoding="utf-8-sig")
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




def figura_serie_mlp(df_test):
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
        name="Calamar predicho - MLP",
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





def figura_serie_rnn(df_test):
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
        name="Calamar predicho - RNN",
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


def figura_serie_lstm(df_test):
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
        name="Calamar predicho - LSTM",
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


def figura_serie_cnn(df_test):
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
        name="Calamar predicho - CNN",
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




def figura_particion_temporal_mlp(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para MLP sin duplicar toda la lógica
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción MLP"
            trace.hovertemplate = (
                "<b>Predicción MLP</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>MLP"

    return fig





def figura_particion_temporal_rnn(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para RNN sin duplicar toda la lógica
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción RNN"
            trace.hovertemplate = (
                "<b>Predicción RNN</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>RNN"

    return fig


def figura_particion_temporal_lstm(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para LSTM sin duplicar toda la lógica
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción LSTM"
            trace.hovertemplate = (
                "<b>Predicción LSTM</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>LSTM"

    return fig


def figura_particion_temporal_cnn(metadata, df_serie, df_test):
    fig = figura_particion_temporal_svr(metadata, df_serie, df_test)

    # Ajustar etiquetas del modelo para CNN sin duplicar toda la lógica
    for trace in fig.data:
        if trace.name == "Predicción SVR":
            trace.name = "Predicción CNN"
            trace.hovertemplate = (
                "<b>Predicción CNN</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            )

    for anot in fig.layout.annotations:
        if getattr(anot, "text", None) == "observado<br>SVR":
            anot.text = "observado<br>CNN"

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




def layout_mlp_calamar():
    metadata, df_test = cargar_resultados_mlp_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_mlp(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_mlp(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "hidden_layer_sizes", "Valor": best_params["hidden_layer_sizes"]},
        {"Parámetro": "alpha", "Valor": best_params["alpha"]},
        {"Parámetro": "learning_rate_init", "Valor": best_params["learning_rate_init"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Multi-Layer Perceptron (MLP) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y MLPRegressor, aplicado a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
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
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros del MLPRegressor. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
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
                        "filename": "particion_temporal_mlp_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo MLP durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])






def layout_rnn_calamar():
    metadata, df_test = cargar_resultados_rnn_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_rnn(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_rnn(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "units", "Valor": best_params["units"]},
        {"Parámetro": "dropout", "Valor": best_params["dropout"]},
        {"Parámetro": "learning_rate", "Valor": best_params["learning_rate"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Recurrent Neural Network (RNN) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una red recurrente RNN implementada en Keras con capa de normalización, aplicada a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
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
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros de la red RNN. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
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
                        "filename": "particion_temporal_rnn_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo RNN durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])




def layout_lstm_calamar():
    metadata, df_test = cargar_resultados_lstm_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_lstm(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_lstm(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "units", "Valor": best_params["units"]},
        {"Parámetro": "dropout", "Valor": best_params["dropout"]},
        {"Parámetro": "learning_rate", "Valor": best_params["learning_rate"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Long Short-Term Memory (LSTM) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una red recurrente LSTM implementada en Keras con capa de normalización, aplicada a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
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
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros de la red LSTM. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
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
                        "filename": "particion_temporal_lstm_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo LSTM durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])




def layout_cnn_calamar():
    metadata, df_test = cargar_resultados_cnn_calamar()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()

    fig_particion = figura_particion_temporal_cnn(
        metadata,
        df_serie_completa,
        df_test
    )

    fig_serie = figura_serie_cnn(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    fig_acf = figura_acf_residuos(df_test, nlags=60)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata["best_params"]
    df_hiper = pd.DataFrame([
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata['numInputs']} días"},
        {"Parámetro": "filters", "Valor": best_params["filters"]},
        {"Parámetro": "kernel_size", "Valor": best_params["kernel_size"]},
        {"Parámetro": "dropout", "Valor": best_params["dropout"]},
        {"Parámetro": "dense_units", "Valor": best_params["dense_units"]},
        {"Parámetro": "learning_rate", "Valor": best_params["learning_rate"]},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata["ventanas_evaluadas"]))},
        {"Parámetro": "Modelos entrenados en búsqueda", "Valor": metadata["modelos_entrenados_busqueda"]},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Convolutional Neural Network (CNN) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una red convolucional 1D implementada en Keras con capa de normalización, aplicada a la predicción del nivel en la estación Calamar. La configuración final se seleccionó usando MAE como criterio principal de validación y MSE como métrica complementaria.",
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
            html.P("La búsqueda evaluó distintas ventanas de entrada y combinaciones de hiperparámetros de la red CNN. La tabla resume la configuración seleccionada y el tamaño de la búsqueda.", style=estilo_parrafo),
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
                        "filename": "particion_temporal_cnn_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2
                    }
                }
            )
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P("La gráfica compara el nivel observado en Calamar con la predicción del modelo CNN durante el periodo reservado como test externo.", style=estilo_parrafo),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P("El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. Idealmente, los residuos deberían concentrarse alrededor de cero y no mostrar autocorrelación marcada.", style=estilo_parrafo),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


# =====================================
# Comparación general de todos los modelos
# =====================================
MODELOS_COMPARACION = [
    {
        "codigo": "svr_calamar",
        "nombre": "SVR",
        "ruta_metadata": RUTA_METADATA_SVR,
        "ruta_test": RUTA_TEST_SVR,
    },
    {
        "codigo": "knn_calamar",
        "nombre": "KNN",
        "ruta_metadata": RUTA_METADATA_KNN,
        "ruta_test": RUTA_TEST_KNN,
    },
    {
        "codigo": "xarima_calamar",
        "nombre": "XARIMA/ARIMA",
        "ruta_metadata": RUTA_METADATA_XARIMA,
        "ruta_test": RUTA_TEST_XARIMA,
    },
    {
        "codigo": "dt_calamar",
        "nombre": "Árbol de Decisión",
        "ruta_metadata": RUTA_METADATA_DT,
        "ruta_test": RUTA_TEST_DT,
    },
    {
        "codigo": "lasso_calamar",
        "nombre": "Lasso",
        "ruta_metadata": RUTA_METADATA_LASSO,
        "ruta_test": RUTA_TEST_LASSO,
    },
    {
        "codigo": "ridge_calamar",
        "nombre": "Ridge",
        "ruta_metadata": RUTA_METADATA_RIDGE,
        "ruta_test": RUTA_TEST_RIDGE,
    },
    {
        "codigo": "rf_calamar",
        "nombre": "Random Forest",
        "ruta_metadata": RUTA_METADATA_RF,
        "ruta_test": RUTA_TEST_RF,
    },
    {
        "codigo": "xgb_calamar",
        "nombre": "XGBoost",
        "ruta_metadata": RUTA_METADATA_XGB,
        "ruta_test": RUTA_TEST_XGB,
    },
    {
        "codigo": "mlp_calamar",
        "nombre": "MLP",
        "ruta_metadata": RUTA_METADATA_MLP,
        "ruta_test": RUTA_TEST_MLP,
    },
    {
        "codigo": "rnn_calamar",
        "nombre": "RNN",
        "ruta_metadata": RUTA_METADATA_RNN,
        "ruta_test": RUTA_TEST_RNN,
    },
    {
        "codigo": "lstm_calamar",
        "nombre": "LSTM",
        "ruta_metadata": RUTA_METADATA_LSTM,
        "ruta_test": RUTA_TEST_LSTM,
    },
    {
        "codigo": "cnn_calamar",
        "nombre": "CNN",
        "ruta_metadata": RUTA_METADATA_CNN,
        "ruta_test": RUTA_TEST_CNN,
    },
]


COLOR_MAPE = AZUL
COLOR_HEATMAP = [[0, "#EAF1F8"], [0.5, CELESTE], [1, AZUL]]


def calcular_mape(y_real, y_pred):
    y_real = pd.to_numeric(y_real, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    mask = y_real.notna() & y_pred.notna() & (y_real != 0)

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100


def _formatear_ventana_comparacion(metadata):
    """Obtiene la ventana seleccionada usando los nombres nuevos y antiguos del metadata."""
    ventana = metadata.get("numInputs_seleccionado", None)

    if ventana is None:
        best_params = metadata.get("best_params", {})
        if isinstance(best_params, dict):
            ventana = best_params.get("numInputs_cv", None)

    if ventana is None:
        ventana = metadata.get("numInputs", None)

    if ventana is None:
        ventana = metadata.get("train_size", None)

    if ventana is None:
        ventanas = metadata.get("numInputs_cv_list", metadata.get("ventanas_evaluadas", None))
        if isinstance(ventanas, (list, tuple)) and len(ventanas) > 0:
            ventana = ventanas[0]

    if ventana is None:
        return "N/A"

    if isinstance(ventana, (int, float)) and not pd.isna(ventana):
        return f"{int(ventana)} días"

    ventana_txt = str(ventana).strip()
    if ventana_txt.upper() == "N/A" or ventana_txt == "":
        return "N/A"
    if "día" not in ventana_txt.lower() and "dia" not in ventana_txt.lower():
        return f"{ventana_txt} días"
    return ventana_txt


def _producto_grid_hiperparametros(grid):
    """Calcula el número de combinaciones de un diccionario de hiperparámetros."""
    if not isinstance(grid, dict) or len(grid) == 0:
        return None

    total = 1
    for valores in grid.values():
        if isinstance(valores, (list, tuple, set)):
            total *= max(1, len(valores))
        else:
            total *= 1
    return total


def _resolver_ruta_desde_metadata(ruta_metadata, valor_ruta):
    """Resuelve archivos referenciados en metadata, incluso si vienen con ruta absoluta de Windows."""
    if not valor_ruta:
        return None

    nombre_archivo = Path(str(valor_ruta).replace("\\", "/")).name
    if not nombre_archivo:
        return None

    candidatos = []
    ruta_metadata = Path(ruta_metadata)

    if ruta_metadata.parent:
        candidatos.append(ruta_metadata.parent / nombre_archivo)

    candidatos.append(Path("Resultados") / nombre_archivo)

    for ruta in candidatos:
        if ruta.exists():
            return ruta

    resultados = Path("Resultados")
    if resultados.exists():
        encontrados = list(resultados.rglob(nombre_archivo))
        if encontrados:
            return encontrados[0]

    return None


def _calcular_modelos_entrenados_comparacion(metadata, ruta_metadata):
    """Obtiene o estima los modelos entrenados para la tabla de la opción Todos."""
    for clave in [
        "modelos_entrenados_busqueda",
        "n_modelos_entrenados",
        "total_modelos_entrenados",
        "modelos_entrenados",
    ]:
        valor = metadata.get(clave, None)
        if valor is not None and not pd.isna(valor):
            return valor

    grid = metadata.get("grid_hiperparametros", None)
    total_grid = _producto_grid_hiperparametros(grid)
    if total_grid is not None:
        ventanas = metadata.get(
            "ventanas_evaluadas",
            metadata.get("numInputs_cv_list", metadata.get("numInputs_list", []))
        )
        if isinstance(ventanas, (list, tuple, set)) and len(ventanas) > 0:
            total_grid *= len(ventanas)
        return int(total_grid)

    ruta_resultados = _resolver_ruta_desde_metadata(
        ruta_metadata,
        metadata.get("archivo_resultados", None),
    )

    if ruta_resultados is not None and Path(ruta_resultados).exists():
        try:
            df_resultados = pd.read_csv(
                ruta_resultados,
                sep=None,
                engine="python",
                encoding="utf-8-sig",
            )
            return int(len(df_resultados))
        except Exception:
            return "N/A"

    return "N/A"


def cargar_comparacion_modelos():
    registros = []
    predicciones = {}

    for spec in MODELOS_COMPARACION:
        with open(spec["ruta_metadata"], "r", encoding="utf-8") as f:
            metadata = json.load(f)

        df_test = pd.read_csv(
            spec["ruta_test"],
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )
        df_test.columns = df_test.columns.str.strip()

        if "Fecha" in df_test.columns:
            df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

        df_test["Calamar_real"] = pd.to_numeric(df_test["Calamar_real"], errors="coerce")
        df_test["Calamar_predicho"] = pd.to_numeric(df_test["Calamar_predicho"], errors="coerce")

        if "Residuo" not in df_test.columns:
            df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]
        else:
            df_test["Residuo"] = pd.to_numeric(df_test["Residuo"], errors="coerce")

        df_test["Error absoluto"] = np.abs(df_test["Calamar_real"] - df_test["Calamar_predicho"])

        mae = float(metadata["MAE_test_externo"])
        mse = float(metadata["MSE_test_externo"])
        rmse = float(np.sqrt(mse))
        mape = float(calcular_mape(df_test["Calamar_real"], df_test["Calamar_predicho"]))

        ventana = _formatear_ventana_comparacion(metadata)
        modelos_entrenados = _calcular_modelos_entrenados_comparacion(
            metadata,
            spec["ruta_metadata"],
        )

        registros.append({
            "Modelo": spec["nombre"],
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Ventana": ventana,
            "Modelos entrenados": modelos_entrenados,
        })

        columnas_pred = ["Fecha", "Calamar_real", "Calamar_predicho", "Residuo", "Error absoluto"]
        predicciones[spec["nombre"]] = df_test[columnas_pred].dropna().copy()

    df_metricas = pd.DataFrame(registros)
    df_metricas = df_metricas.sort_values("MAE", ascending=True).reset_index(drop=True)
    df_metricas.insert(0, "Ranking", np.arange(1, len(df_metricas) + 1))

    return df_metricas, predicciones

def tabla_comparativa_modelos(df_metricas):
    df_tabla = df_metricas.copy()
    for col in ["MAE", "MSE", "RMSE", "MAPE"]:
        df_tabla[col] = df_tabla[col].round(4)

    modelo_mejor = df_tabla.iloc[0]["Modelo"]
    modelo_peor = df_tabla.iloc[-1]["Modelo"]

    return dash_table.DataTable(
        data=df_tabla.to_dict("records"),
        columns=[
            {"name": "Ranking", "id": "Ranking"},
            {"name": "Modelo", "id": "Modelo"},
            {"name": "MAE", "id": "MAE", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "MSE", "id": "MSE", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "RMSE", "id": "RMSE", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "MAPE [%]", "id": "MAPE", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Ventana seleccionada", "id": "Ventana"},
            {"name": "Modelos entrenados", "id": "Modelos entrenados"},
        ],
        page_size=12,
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
        style_data_conditional=[
            {
                "if": {"filter_query": f'{{Modelo}} = "{modelo_mejor}"'},
                "backgroundColor": "#DDF3EA",
                "color": "#006B4F",
                "fontWeight": "bold",
            },
            {
                "if": {"filter_query": f'{{Modelo}} = "{modelo_peor}"'},
                "backgroundColor": "#F9DADA",
                "color": "#8A1F1F",
                "fontWeight": "bold",
            },
        ],
    )


def figura_mosaico_metricas(df_metricas):
    df_plot = df_metricas.sort_values("MAE", ascending=True).copy()

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=[
            "MAE por modelo",
            "RMSE por modelo",
            "MAPE [%] por modelo",
        ],
        vertical_spacing=0.20,
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["Modelo"],
            y=df_plot["MAE"],
            text=df_plot["MAE"].round(3),
            textposition="outside",
            marker_color=AZUL_MED,
            hovertemplate="<b>Modelo:</b> %{x}<br><b>MAE:</b> %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["Modelo"],
            y=df_plot["RMSE"],
            text=df_plot["RMSE"].round(3),
            textposition="outside",
            marker_color=CELESTE,
            hovertemplate="<b>Modelo:</b> %{x}<br><b>RMSE:</b> %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["Modelo"],
            y=df_plot["MAPE"],
            text=df_plot["MAPE"].round(3),
            textposition="outside",
            marker_color=COLOR_MAPE,
            hovertemplate="<b>Modelo:</b> %{x}<br><b>MAPE:</b> %{y:.4f} %<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=850,
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=70, r=40, t=90, b=95),
    )

    fig.update_xaxes(tickangle=-30, showgrid=False, row=1, col=1)
    fig.update_xaxes(tickangle=-30, showgrid=False, row=1, col=2)
    fig.update_xaxes(tickangle=-30, showgrid=False, row=2, col=1)

    fig.update_yaxes(title_text="MAE", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="RMSE", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=1, col=2)
    fig.update_yaxes(title_text="MAPE [%]", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=2, col=1)

    for anot in fig.layout.annotations:
        anot.font = dict(family=FUENTE, size=16, color=AZUL)

    return fig


def figura_ranking_mae_modelos(df_metricas):
    df_plot = df_metricas.sort_values("MAE", ascending=False).copy()
    mejor_mae = df_metricas["MAE"].min()
    df_plot["Diferencia relativa [%]"] = ((df_plot["MAE"] - mejor_mae) / mejor_mae) * 100

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Ranking horizontal por MAE",
            "Diferencia relativa frente al mejor MAE",
        ],
        horizontal_spacing=0.18,
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["MAE"],
            y=df_plot["Modelo"],
            orientation="h",
            text=df_plot["MAE"].round(3),
            textposition="outside",
            marker_color=AZUL_MED,
            hovertemplate="<b>Modelo:</b> %{y}<br><b>MAE:</b> %{x:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["Diferencia relativa [%]"],
            y=df_plot["Modelo"],
            orientation="h",
            text=df_plot["Diferencia relativa [%]"].round(1).astype(str) + " %",
            textposition="outside",
            marker_color=CELESTE,
            hovertemplate=(
                "<b>Modelo:</b> %{y}<br>"
                "<b>Diferencia frente al mejor MAE:</b> %{x:.2f} %"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=620,
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=120, r=60, t=85, b=60),
    )

    fig.update_xaxes(title_text="MAE", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="Diferencia relativa [%]", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=2)

    for anot in fig.layout.annotations:
        anot.font = dict(family=FUENTE, size=16, color=AZUL)

    return fig


def figura_dispersion_metricas(df_metricas):
    df_plot = df_metricas.sort_values("MAE", ascending=True).copy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Relación MAE vs RMSE",
            "Relación MAE vs MAPE [%]",
        ],
        horizontal_spacing=0.14,
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["MAE"],
            y=df_plot["RMSE"],
            mode="markers+text",
            text=df_plot["Modelo"],
            textposition="top center",
            marker=dict(
                size=13,
                color=df_plot["Ranking"],
                colorscale="Blues",
                showscale=False,
                line=dict(width=1, color=AZUL),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "<b>MAE:</b> %{x:.4f}<br>"
                "<b>RMSE:</b> %{y:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["MAE"],
            y=df_plot["MAPE"],
            mode="markers+text",
            text=df_plot["Modelo"],
            textposition="top center",
            marker=dict(
                size=13,
                color=df_plot["Ranking"],
                colorscale="Blues",
                showscale=False,
                line=dict(width=1, color=AZUL),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "<b>MAE:</b> %{x:.4f}<br>"
                "<b>MAPE:</b> %{y:.4f} %"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=560,
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=70, r=40, t=85, b=65),
    )

    fig.update_xaxes(title_text="MAE", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="MAE", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=1, col=2)
    fig.update_yaxes(title_text="RMSE", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="MAPE [%]", showgrid=True, gridcolor="#D9E2EF", zeroline=False, row=1, col=2)

    for anot in fig.layout.annotations:
        anot.font = dict(family=FUENTE, size=16, color=AZUL)

    return fig


def figura_boxplot_error_absoluto(predicciones, df_metricas):
    modelos_ordenados = df_metricas["Modelo"].tolist()
    fig = go.Figure()

    for modelo in modelos_ordenados:
        df_pred = predicciones[modelo]
        fig.add_trace(
            go.Box(
                y=df_pred["Error absoluto"],
                name=modelo,
                boxmean=True,
                marker_color=AZUL_MED,
                line_color=AZUL,
                hovertemplate=(
                    f"<b>{modelo}</b><br>"
                    "<b>Error absoluto:</b> %{y:.3f} cm"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=None,
        yaxis_title="Error absoluto [cm]",
        xaxis_title="Modelo",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=70, r=40, t=50, b=120),
        height=560,
        showlegend=False,
    )

    fig.update_xaxes(tickangle=-30, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)

    return fig


def figura_heatmap_metricas_normalizadas(df_metricas):
    metricas = ["MAE", "RMSE", "MAPE"]
    df_norm = df_metricas[["Modelo"] + metricas].copy()

    for metrica in metricas:
        minimo = df_norm[metrica].min()
        maximo = df_norm[metrica].max()
        if maximo == minimo:
            df_norm[metrica] = 0
        else:
            df_norm[metrica] = (df_norm[metrica] - minimo) / (maximo - minimo)

    fig = go.Figure(
        data=go.Heatmap(
            z=df_norm[metricas].values,
            x=["MAE", "RMSE", "MAPE [%]"],
            y=df_norm["Modelo"],
            colorscale=COLOR_HEATMAP,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Error normalizado"),
            hovertemplate=(
                "<b>Modelo:</b> %{y}<br>"
                "<b>Métrica:</b> %{x}<br>"
                "<b>Valor normalizado:</b> %{z:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Métrica",
        yaxis_title="Modelo",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=130, r=60, t=50, b=70),
        height=520,
    )

    return fig


def figura_real_vs_predicho_mejores(predicciones, df_metricas, n_modelos=5):
    mejores_modelos = df_metricas.sort_values("MAE", ascending=True).head(n_modelos)["Modelo"].tolist()
    fig = go.Figure()

    min_val = np.inf
    max_val = -np.inf

    for modelo in mejores_modelos:
        df_pred = predicciones[modelo]
        min_val = min(min_val, df_pred["Calamar_real"].min(), df_pred["Calamar_predicho"].min())
        max_val = max(max_val, df_pred["Calamar_real"].max(), df_pred["Calamar_predicho"].max())

        fig.add_trace(
            go.Scatter(
                x=df_pred["Calamar_real"],
                y=df_pred["Calamar_predicho"],
                mode="markers",
                name=modelo,
                marker=dict(size=7, opacity=0.65),
                hovertemplate=(
                    f"<b>{modelo}</b><br>"
                    "<b>Real:</b> %{x:.2f} cm<br>"
                    "<b>Predicho:</b> %{y:.2f} cm"
                    "<extra></extra>"
                ),
            )
        )

    margen = 0.04 * (max_val - min_val) if np.isfinite(max_val - min_val) and max_val != min_val else 1
    min_linea = min_val - margen
    max_linea = max_val + margen

    fig.add_trace(
        go.Scatter(
            x=[min_linea, max_linea],
            y=[min_linea, max_linea],
            mode="lines",
            name="Línea 1:1",
            line=dict(color=AZUL, width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Nivel observado [cm]",
        yaxis_title="Nivel predicho [cm]",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
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
        margin=dict(l=70, r=40, t=80, b=70),
        height=580,
    )

    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False, range=[min_linea, max_linea])
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False, range=[min_linea, max_linea])

    return fig


def figura_series_todos_modelos(predicciones, df_metricas):
    fig = go.Figure()

    primer_modelo = df_metricas.iloc[0]["Modelo"]
    df_base = predicciones[primer_modelo]

    fig.add_trace(go.Scatter(
        x=df_base["Fecha"],
        y=df_base["Calamar_real"],
        mode="lines",
        name="Calamar real",
        line=dict(color=AZUL, width=3),
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Nivel real:</b> %{y:.2f} cm<br><extra></extra>",
    ))

    modelos_visibles = set(df_metricas.head(3)["Modelo"])

    for modelo, df_pred in predicciones.items():
        fig.add_trace(go.Scatter(
            x=df_pred["Fecha"],
            y=df_pred["Calamar_predicho"],
            mode="lines",
            name=f"Predicción {modelo}",
            line=dict(width=2, dash="dash"),
            visible=True if modelo in modelos_visibles else "legendonly",
            hovertemplate=(
                f"<b>Predicción {modelo}</b><br>"
                "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                "<b>Nivel predicho:</b> %{y:.2f} cm<br>"
                "<extra></extra>"
            ),
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
        margin=dict(l=70, r=40, t=80, b=60),
        height=560,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    return fig



MAPA_NOMBRES_DM = {
    "RandomForest": "Random Forest",
    "DecisionTree": "Árbol de Decisión",
    "XARIMA": "XARIMA/ARIMA",
}


def leer_matriz_pvalues_dm(rutas_posibles):
    ruta_encontrada = None

    for ruta in rutas_posibles:
        if ruta.exists():
            ruta_encontrada = ruta
            break

    if ruta_encontrada is None:
        rutas_txt = ", ".join(str(ruta) for ruta in rutas_posibles)
        raise FileNotFoundError(f"No se encontró la matriz de p-valores DM. Rutas revisadas: {rutas_txt}")

    if ruta_encontrada.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(ruta_encontrada)
    else:
        df = pd.read_csv(
            ruta_encontrada,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )

    df.columns = df.columns.astype(str).str.strip()

    primera_columna = df.columns[0]
    if primera_columna.startswith("Unnamed") or primera_columna == "":
        df = df.set_index(primera_columna)
    elif primera_columna not in df.columns[1:].tolist():
        df = df.set_index(primera_columna)

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    df = df.rename(index=MAPA_NOMBRES_DM, columns=MAPA_NOMBRES_DM)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def figura_heatmap_pvalues_dm(df_pvalues):
    z = df_pvalues.copy()

    # Crear matriz de texto como object para permitir strings vacíos
    texto = df_pvalues.copy().astype(object)

    for i in range(min(z.shape)):
        z.iat[i, i] = np.nan
        texto.iat[i, i] = ""

    texto = texto.map(
        lambda valor: "" if pd.isna(valor) or valor == "" else f"{float(valor):.3g}"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=df_pvalues.columns,
            y=df_pvalues.index,
            text=texto.values,
            texttemplate="%{text}",
            textfont=dict(size=10, color=AZUL),
            zmin=0,
            zmax=1,
            colorscale=[
                [0.00, "#B23A48"],
                [0.05, "#F4D7D7"],
                [0.25, "#EAF1F8"],
                [1.00, AZUL],
            ],
            colorbar=dict(
                title="p-valor",
                tickfont=dict(family=FUENTE, size=12, color=AZUL),
            ),
            hovertemplate=(
                "<b>Modelo fila:</b> %{y}<br>"
                "<b>Modelo columna:</b> %{x}<br>"
                "<b>p-valor:</b> %{z:.4g}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Modelo",
        yaxis_title="Modelo",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=140, r=80, t=40, b=120),
        height=720,
    )

    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)

    return fig

MAPA_NOMBRES_MANNWHITNEY = {
    "RandomForest": "Random Forest",
    "DecisionTree": "Árbol de Decisión",
    "XARIMA": "XARIMA/ARIMA",
}


def leer_archivo_mannwhitney_modelos(rutas_posibles, descripcion):
    ruta_encontrada = None

    for ruta in rutas_posibles:
        if ruta.exists():
            ruta_encontrada = ruta
            break

    if ruta_encontrada is None:
        rutas_txt = ", ".join(str(ruta) for ruta in rutas_posibles)
        raise FileNotFoundError(f"No se encontró el archivo de {descripcion}. Rutas revisadas: {rutas_txt}")

    if ruta_encontrada.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(ruta_encontrada)
    else:
        df = pd.read_csv(
            ruta_encontrada,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )

    df.columns = df.columns.astype(str).str.strip()
    return df


def figura_heatmap_errores_absolutos_mannwhitney(df_errores):
    df = df_errores.copy()
    df.columns = df.columns.astype(str).str.strip()

    columna_fecha = None
    for col in df.columns:
        if col.lower() == "fecha":
            columna_fecha = col
            break

    if columna_fecha is not None:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
        df = df.set_index(columna_fecha)
        df.index = df.index.strftime("%Y-%m-%d")

    df = df.rename(columns=MAPA_NOMBRES_MANNWHITNEY)
    df = df.apply(pd.to_numeric, errors="coerce")

    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            zmin=0,
            zmax=float(np.nanmax(df.values)) if np.isfinite(np.nanmax(df.values)) else 1,
            colorscale=[
                [0.00, "#B23A48"],
                [0.05, "#F4D7D7"],
                [0.25, "#EAF1F8"],
                [1.00, AZUL],
            ],
            colorbar=dict(
                title="Error absoluto",
                tickfont=dict(family=FUENTE, size=12, color=AZUL),
            ),
            hovertemplate=(
                "<b>Fecha:</b> %{y}<br>"
                "<b>Modelo:</b> %{x}<br>"
                "<b>Error absoluto:</b> %{z:.4g}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Modelo",
        yaxis_title="Fecha",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=120, r=80, t=40, b=120),
        height=760,
    )

    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def figura_heatmap_mannwhitney_svr_vs_modelos(df_resultados):
    df = df_resultados.copy()
    df.columns = df.columns.astype(str).str.strip()

    columna_modelo = "modelo_comparado"
    if columna_modelo not in df.columns:
        raise ValueError("No se encontró la columna 'modelo_comparado' en resultados_mannwhitney_svr_vs_modelos.")

    columna_pvalor = "p_valor_ajustado_holm"
    if columna_pvalor not in df.columns:
        columna_pvalor = "p_valor"

    if columna_pvalor not in df.columns:
        raise ValueError("No se encontró la columna de p-valor en resultados_mannwhitney_svr_vs_modelos.")

    df[columna_modelo] = df[columna_modelo].astype(str).str.strip().replace(MAPA_NOMBRES_MANNWHITNEY)
    df[columna_pvalor] = pd.to_numeric(df[columna_pvalor], errors="coerce")

    modelos = df[columna_modelo].tolist()
    valores = df[columna_pvalor].values.reshape(1, -1)
    texto = np.array([[("" if pd.isna(v) else f"{float(v):.3g}") for v in df[columna_pvalor].values]])

    fig = go.Figure(
        data=go.Heatmap(
            z=valores,
            x=modelos,
            y=["SVR"],
            text=texto,
            texttemplate="%{text}",
            textfont=dict(size=10, color=AZUL),
            zmin=0,
            zmax=1,
            colorscale=[
                [0.00, "#B23A48"],
                [0.05, "#F4D7D7"],
                [0.25, "#EAF1F8"],
                [1.00, AZUL],
            ],
            colorbar=dict(
                title="p-valor",
                tickfont=dict(family=FUENTE, size=12, color=AZUL),
            ),
            hovertemplate=(
                "<b>Referencia:</b> SVR<br>"
                "<b>Modelo comparado:</b> %{x}<br>"
                "<b>p-valor:</b> %{z:.4g}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Modelo comparado",
        yaxis_title="Modelo de referencia",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=120, r=80, t=40, b=120),
        height=360,
    )

    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def figura_heatmap_mannwhitney_todos_contra_todos(df_resultados):
    df = df_resultados.copy()
    df.columns = df.columns.astype(str).str.strip()

    columna_a = "modelo_a"
    columna_b = "modelo_b"
    if columna_a not in df.columns or columna_b not in df.columns:
        raise ValueError("No se encontraron las columnas 'modelo_a' y 'modelo_b' en resultados_mannwhitney_todos_contra_todos.")

    columna_pvalor = "p_valor_ajustado_holm"
    if columna_pvalor not in df.columns:
        columna_pvalor = "p_valor"

    if columna_pvalor not in df.columns:
        raise ValueError("No se encontró la columna de p-valor en resultados_mannwhitney_todos_contra_todos.")

    df[columna_a] = df[columna_a].astype(str).str.strip().replace(MAPA_NOMBRES_MANNWHITNEY)
    df[columna_b] = df[columna_b].astype(str).str.strip().replace(MAPA_NOMBRES_MANNWHITNEY)
    df[columna_pvalor] = pd.to_numeric(df[columna_pvalor], errors="coerce")

    modelos = []
    for modelo in list(df[columna_a]) + list(df[columna_b]):
        if modelo not in modelos:
            modelos.append(modelo)

    matriz = pd.DataFrame(np.nan, index=modelos, columns=modelos)

    for _, fila in df.iterrows():
        modelo_a = fila[columna_a]
        modelo_b = fila[columna_b]
        pvalor = fila[columna_pvalor]
        matriz.loc[modelo_a, modelo_b] = pvalor
        matriz.loc[modelo_b, modelo_a] = pvalor

    for i in range(min(matriz.shape)):
        matriz.iat[i, i] = np.nan

    texto = matriz.copy().astype(object)

    texto = texto.map(
        lambda valor: "" if pd.isna(valor) else f"{float(valor):.3g}"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matriz.values,
            x=matriz.columns,
            y=matriz.index,
            text=texto.values,
            texttemplate="%{text}",
            textfont=dict(size=10, color=AZUL),
            zmin=0,
            zmax=1,
            colorscale=[
                [0.00, "#B23A48"],
                [0.05, "#F4D7D7"],
                [0.25, "#EAF1F8"],
                [1.00, AZUL],
            ],
            colorbar=dict(
                title="p-valor",
                tickfont=dict(family=FUENTE, size=12, color=AZUL),
            ),
            hovertemplate=(
                "<b>Modelo A:</b> %{y}<br>"
                "<b>Modelo B:</b> %{x}<br>"
                "<b>p-valor:</b> %{z:.4g}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Modelo",
        yaxis_title="Modelo",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=140, r=80, t=40, b=120),
        height=720,
    )

    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)

    return fig

def layout_todos_modelos():
    df_metricas, predicciones = cargar_comparacion_modelos()

    mejor_mae = df_metricas.loc[df_metricas["MAE"].idxmin()]
    mejor_rmse = df_metricas.loc[df_metricas["RMSE"].idxmin()]
    mejor_mape = df_metricas.loc[df_metricas["MAPE"].idxmin()]

    fig_mosaico_metricas = figura_mosaico_metricas(df_metricas)
    fig_ranking = figura_ranking_mae_modelos(df_metricas)
    fig_dispersion = figura_dispersion_metricas(df_metricas)
    fig_boxplot_error = figura_boxplot_error_absoluto(predicciones, df_metricas)
    fig_heatmap_metricas = figura_heatmap_metricas_normalizadas(df_metricas)
    fig_real_predicho = figura_real_vs_predicho_mejores(predicciones, df_metricas, n_modelos=5)
    fig_series = figura_series_todos_modelos(predicciones, df_metricas)
    df_dm_absoluta = leer_matriz_pvalues_dm(RUTAS_DM_ABSOLUTA)
    df_dm_cuadratica = leer_matriz_pvalues_dm(RUTAS_DM_CUADRATICA)
    fig_dm_absoluta = figura_heatmap_pvalues_dm(df_dm_absoluta)
    fig_dm_cuadratica = figura_heatmap_pvalues_dm(df_dm_cuadratica)
    df_mw_errores_absolutos = leer_archivo_mannwhitney_modelos(
        RUTAS_MW_ERRORES_ABSOLUTOS,
        "matriz de errores absolutos por modelo",
    )
    df_mw_svr_vs_modelos = leer_archivo_mannwhitney_modelos(
        RUTAS_MW_SVR_VS_MODELOS,
        "resultados Mann-Whitney SVR vs modelos",
    )
    df_mw_todos_contra_todos = leer_archivo_mannwhitney_modelos(
        RUTAS_MW_TODOS_CONTRA_TODOS,
        "resultados Mann-Whitney todos contra todos",
    )
    fig_mw_errores_absolutos = figura_heatmap_errores_absolutos_mannwhitney(df_mw_errores_absolutos)
    fig_mw_svr_vs_modelos = figura_heatmap_mannwhitney_svr_vs_modelos(df_mw_svr_vs_modelos)
    fig_mw_todos_contra_todos = figura_heatmap_mannwhitney_todos_contra_todos(df_mw_todos_contra_todos)

    texto_conclusion = (
    "Con el fin de evaluar la eficacia de cada modelo para predecir el nivel del río, se hizo uso de un test externo correspondiente "
    "a un año de datos 2025 y se midieron las métricas de MAE, RMSE y MAPE. Se observa que el modelo SVR presentó el mejor desempeño "
    "global frente a los demás modelos analizados. Este obtuvo el menor valor de MAE = 1.604, lo que indica que, en promedio, sus "
    "predicciones se desviaron menos de los valores reales del nivel en Calamar. Asimismo, este modelo alcanzó el menor RMSE = 2.144 y "
    "el menor MAPE = 0.304 %, lo que refleja un error porcentual relativo muy bajo. En conjunto, estos resultados sugieren que el modelo "
    "SVR logró una mejor capacidad predictiva en el periodo de prueba externo, por lo que puede considerarse como el más apto para la "
    "tarea de predicción en la estación de estudio."
    )

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Comparación general de modelos - Calamar", style=estilo_titulo),
            html.P(
                "Esta sección compara el desempeño de todos los modelos implementados para la predicción "
                "del nivel en la estación Calamar. La comparación se realiza sobre el mismo periodo de test "
                "externo, reservado como conjunto no utilizado durante la selección de hiperparámetros.",
                style=estilo_parrafo,
            ),
            html.P(
                "Se presentan las métricas MAE, MSE, RMSE y MAPE. En todos los casos, valores menores "
                "indican mejor desempeño predictivo.",
                style=estilo_parrafo,
            ),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("Mejor MAE", mejor_mae["Modelo"], f"MAE = {mejor_mae['MAE']:.3f}"),
            tarjeta_metrica("Mejor RMSE", mejor_rmse["Modelo"], f"RMSE = {mejor_rmse['RMSE']:.3f}"),
            tarjeta_metrica("Mejor MAPE", mejor_mape["Modelo"], f"MAPE = {mejor_mape['MAPE']:.3f} %"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Tabla comparativa de métricas", style=estilo_titulo),
            html.P(
                "La tabla resume el desempeño de cada modelo en el test externo. La fila verde "
                "resalta el modelo con menor MAE y la fila roja el modelo con mayor MAE.",
                style=estilo_parrafo,
            ),
            tabla_comparativa_modelos(df_metricas),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Comparación gráfica de métricas", style=estilo_titulo),
            html.P(
                "Las gráficas comparan el desempeño de los modelos en el test externo. "
                "Valores menores de MAE, RMSE y MAPE indican mejor desempeño predictivo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_mosaico_metricas,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "comparacion_metricas_modelos",
                        "height": 1400,
                        "width": 1600,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Análisis gráfico del ranking", style=estilo_titulo),
            html.P(
                "El ranking se representa mediante barras horizontales ordenadas por MAE. "
                "También se muestra la diferencia relativa de cada modelo frente al mejor desempeño, "
                "lo que permite ver cuánto aumenta el error respecto al modelo con menor MAE.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_ranking,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "ranking_modelos_mae",
                        "height": 1000,
                        "width": 1600,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Relación entre métricas de error", style=estilo_titulo),
            html.P(
                "Los diagramas de dispersión permiten comparar simultáneamente las métricas. "
                "Los modelos ubicados hacia la parte inferior izquierda presentan menores errores "
                "en las dos métricas comparadas.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_dispersion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "relacion_metricas_modelos",
                        "height": 900,
                        "width": 1600,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Distribución del error absoluto", style=estilo_titulo),
            html.P(
                "El boxplot muestra la dispersión de los errores absolutos por modelo. "
                "Permite identificar modelos con errores concentrados y modelos con valores extremos.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_boxplot_error,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "boxplot_error_absoluto_modelos",
                        "height": 900,
                        "width": 1600,
                        "scale": 2,
                    },
                },
            ),
        ]),
            
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Observado vs predicho en los mejores modelos", style=estilo_titulo),
            html.P(
                "La gráfica compara los niveles observados contra los predichos para los cinco mejores modelos según MAE. "
                "Mientras más cerca estén los puntos de la línea 1:1, mejor es la correspondencia entre predicción y observación.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_real_predicho,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "observado_vs_predicho_mejores_modelos",
                        "height": 1000,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicciones de los modelos", style=estilo_titulo),
            html.P(
                "La gráfica muestra la serie observada y las predicciones de los modelos durante el test externo. "
                "Para evitar saturación visual, inicialmente se muestran los tres mejores modelos según MAE; "
                "los demás pueden activarse desde la leyenda de Plotly.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_series,
                config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Interpretación comparativa", style=estilo_titulo),
            html.P(texto_conclusion, style=estilo_parrafo),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Mapas de calor de p-valores Diebold-Mariano", style=estilo_titulo),
            html.P(
                "El test de Diebold–Mariano se aplicó para comparar la capacidad predictiva de los modelos sobre el mismo periodo "
                "de test externo, usando pérdida absoluta y pérdida cuadrática, asociadas al MAE y al MSE, respectivamente. "
                "Los resultados mostraron que SVR obtuvo el menor error promedio en el test externo, sin embargo, su diferencia "
                "con Random Forest no fue estadísticamente significativa en ninguna de las dos funciones de pérdida. Esto indica que, "
                "aunque SVR presentó el mejor desempeño promedio, Random Forest tuvo una capacidad predictiva estadísticamente comparable. "
                "En contraste, varios modelos con mayores errores, como MLP, LSTM, RNN y XARIMA, sí presentaron diferencias significativas "
                "frente a los modelos de mejor desempeño, evidenciando una menor precisión relativa en el periodo evaluado.",
                style=estilo_parrafo,
            ),
            
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr",
                    "gap": "28px",
                    "alignItems": "stretch",
                },
                children=[
                    html.Div(children=[
                        html.H3("Diferencia absoluta", style={**estilo_titulo, "fontSize": "18px"}),
                        html.P(
                            "Este mapa presenta los p-valores de la prueba Diebold-Mariano calculada sobre "
                            "la diferencia absoluta entre errores. Valores menores que 0.05 indican diferencias "
                            "estadísticamente significativas entre el desempeño predictivo de los modelos comparados.",
                            style=estilo_parrafo,
                        ),
                        dcc.Graph(
                            figure=fig_dm_absoluta,
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                                "displaylogo": False,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "heatmap_pvalues_dm_absoluta",
                                    "height": 1100,
                                    "width": 1400,
                                    "scale": 2,
                                },
                            },
                        ),
                    ]),
                    html.Div(children=[
                        html.H3("Diferencia cuadrática", style={**estilo_titulo, "fontSize": "18px"}),
                        html.P(
                            "Este mapa presenta los p-valores de la prueba Diebold-Mariano calculada sobre "
                            "la diferencia cuadrática entre errores. Valores menores que 0.05 indican diferencias "
                            "estadísticamente significativas entre el desempeño predictivo de los modelos comparados.",
                            style=estilo_parrafo,
                        ),
                        dcc.Graph(
                            figure=fig_dm_cuadratica,
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                                "displaylogo": False,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "heatmap_pvalues_dm_cuadratica",
                                    "height": 1100,
                                    "width": 1400,
                                    "scale": 2,
                                },
                            },
                        ),
                    ]),
                ],
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diferencias entre modelos con el Test Mann-Whitney", style=estilo_titulo),
            html.P(
                "Para comprender la significancia entre las diferencias de los modelos, se optó por aplicar el test Mann-Whitney. "
                "Los valores de P fueron graficados en una matriz, esta muestra que, en la mayoría de comparaciones, los modelos sí "
                "presentan diferencias claras entre sus errores, porque muchos valores de P son menores a 0.001. Esto quiere decir "
                "que no solo hay diferencias en las métricas generales como MAE o RMSE, sino que esas diferencias también se reflejan "
                "en los errores del test externo. Sin embargo, hay algunos grupos de modelos que se comportan de forma parecida, por "
                "ejemplo SVR y Random Forest, donde el valor-p ajustado fue alto, lo que indica que sus errores no son estadísticamente "
                "tan distintos. Algo similar ocurre entre Ridge y Lasso, y entre XGBoost y Decision Tree. En general, la prueba confirma "
                "que SVR fue uno de los modelos más competitivos, pero también muestra que su ventaja frente a algunos modelos cercanos, "
                "como Random Forest, no es tan marcada desde el punto de vista estadístico y se podría optar por modelos con menor costo "
                "computacional.",
                style=estilo_parrafo,
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr",
                    "gap": "28px",
                    "alignItems": "stretch",
                },
                children=[
                    html.Div(children=[
                        html.H3("Mann-Whitney: SVR vs modelos", style={**estilo_titulo, "fontSize": "18px"}),
                        html.P(
                            "Este mapa muestra los p-valores ajustados del test Mann-Whitney al comparar "
                            "la distribución de errores absolutos del SVR frente a los demás modelos.",
                            style=estilo_parrafo,
                        ),
                        dcc.Graph(
                            figure=fig_mw_svr_vs_modelos,
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                                "displaylogo": False,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "heatmap_mannwhitney_svr_vs_modelos",
                                    "height": 700,
                                    "width": 1400,
                                    "scale": 2,
                                },
                            },
                        ),
                    ]),
                    html.Div(children=[
                        html.H3("Mann-Whitney: todos contra todos", style={**estilo_titulo, "fontSize": "18px"}),
                        html.P(
                            "Este mapa muestra los p-valores ajustados del test Mann-Whitney para todas "
                            "las comparaciones pareadas entre modelos. Valores menores que 0.05 indican "
                            "diferencias estadísticamente significativas entre distribuciones de errores.",
                            style=estilo_parrafo,
                        ),
                        dcc.Graph(
                            figure=fig_mw_todos_contra_todos,
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                                "displaylogo": False,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "heatmap_mannwhitney_todos_contra_todos",
                                    "height": 1100,
                                    "width": 1400,
                                    "scale": 2,
                                },
                            },
                        ),
                    ]),
                ],
            ),
        ]),
    ])

def layout_modelos(modelos=None):
    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Modelos implementados", style=estilo_titulo),
            html.P(
                "Debido a la alta multicolinealidad observada entre las series de nivel de las diferentes estaciones, se decidió "
                "construir los modelos utilizando únicamente la información temporal y el nivel registrado en Calamar. Esta "
                "decisión se tomó especialmente para evitar problemas en los modelos lineales, como Ridge y Lasso, en los "
                "que la presencia de variables altamente correlacionadas puede afectar la estabilidad e interpretación de los "
                "coeficientes. De esta forma, se trabajó bajo un esquema univariado, donde el nivel futuro se predice a partir de "
                "la memoria reciente de la propia serie. ",
                style=estilo_parrafo,
            ),
            
            html.P(
                "A continuación seleccione un modelo para visualizar su configuración, validación temporal, métricas, predicciones y diagnóstico de residuos " 
                "en la estación de Calamar.",
                style=estilo_parrafo,
            ),
            
            dcc.Dropdown(
                id="selector-modelo",
                options=[
                    {"label": "Todos", "value": "todos"},
                    {"label": "Máquina de Vectores de Soporte (SVR)", "value": "svr_calamar"},
                    {"label": "K-Vecinos más Cercanos (KNN)", "value": "knn_calamar"},
                    {"label": "XARIMA/ARIMA", "value": "xarima_calamar"},
                    {"label": "Árbol de Decisión", "value": "dt_calamar"},
                    {"label": "Regresión Lasso", "value": "lasso_calamar"},
                    {"label": "Regresión Ridge", "value": "ridge_calamar"},
                    {"label": "Random Forest", "value": "rf_calamar"},
                    {"label": "XGBoost", "value": "xgb_calamar"},
                    {"label": "Multi-Layer Perceptron (MLP)", "value": "mlp_calamar"},
                    {"label": "Recurrent Neural Network (RNN)", "value": "rnn_calamar"},
                    {"label": "Long Short-Term Memory (LSTM)", "value": "lstm_calamar"},
                    {"label": "Convolutional Neural Network (CNN)", "value": "cnn_calamar"},
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
        if modelo == "todos":
            return layout_todos_modelos()

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

        if modelo == "mlp_calamar":
            return layout_mlp_calamar()

        if modelo == "rnn_calamar":
            return layout_rnn_calamar()

        if modelo == "lstm_calamar":
            return layout_lstm_calamar()

        if modelo == "cnn_calamar":
            return layout_cnn_calamar()

        return html.Div(style=estilo_tarjeta, children=[
            html.H2("Modelo no disponible", style=estilo_titulo),
            html.P("La información de este modelo todavía no ha sido cargada.", style=estilo_parrafo),
        ])


# =====================================================================
# ACTUALIZACIÓN SVR MULTIOUTPUT H10 + GRÁFICA GENERAL DE PLIEGUES
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================


def _resolver_archivo_resultados(nombre_archivo, subcarpetas=("SVR", "1_SVR")):
    """Busca el archivo en las rutas más probables sin romper el Dash si cambia la carpeta."""
    candidatos = []
    for sub in subcarpetas:
        candidatos.append(Path("Resultados") / sub / nombre_archivo)
    candidatos.append(Path("Resultados") / nombre_archivo)
    for sub in subcarpetas:
        candidatos.append(Path(sub) / nombre_archivo)

    for ruta in candidatos:
        if ruta.exists():
            return ruta
    return candidatos[0]


# Nuevas rutas para SVR multioutput H10
RUTA_METADATA_SVR = _resolver_archivo_resultados("metadata_modelo_svr_multioutput_h10_calamar.json")
RUTA_TEST_SVR = _resolver_archivo_resultados("test_final_externo_svr_multioutput_h10_calamar.csv")
RUTA_MODELO_SVR = _resolver_archivo_resultados("modelo_svr_multioutput_h10_calamar.joblib")
RUTA_RESUMEN_SVR = _resolver_archivo_resultados("resumen_svr_multioutput_h10_fijo_timeseries_cv_bds.csv")
RUTA_RESULTADOS_CV_SVR = _resolver_archivo_resultados("resultados_svr_multioutput_h10_fijo_timeseries_cv_bds.csv")
RUTA_METRICAS_HORIZONTES_SVR = _resolver_archivo_resultados("metricas_horizontes_1_5_10_svr_multioutput_h10.csv")
RUTA_PLIEGUES_VALIDACION = _resolver_archivo_resultados("plotly_pliegues_reales_mejor_ventana_svr_multioutput_h10.csv")

# Actualizar la entrada SVR de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "svr_calamar":
        _spec["nombre"] = "SVR Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_SVR
        _spec["ruta_test"] = RUTA_TEST_SVR


def _leer_csv_modelos(ruta):
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def cargar_resultados_svr_calamar():
    with open(RUTA_METADATA_SVR, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_modelos(RUTA_TEST_SVR)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_svr_multioutput():
    df_resumen = _leer_csv_modelos(RUTA_RESUMEN_SVR)
    df_resultados_cv = _leer_csv_modelos(RUTA_RESULTADOS_CV_SVR)
    df_metricas_horizontes = _leer_csv_modelos(RUTA_METRICAS_HORIZONTES_SVR)
    return df_resumen, df_resultados_cv, df_metricas_horizontes


def figura_pliegues_validacion():
    fig = go.Figure()

    if not Path(RUTA_PLIEGUES_VALIDACION).exists():
        fig.add_annotation(
            text="No se encontró el archivo de pliegues de validación cruzada temporal.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=360,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    df_pliegues = _leer_csv_modelos(RUTA_PLIEGUES_VALIDACION)

    for col in ["x_inicio", "x_fin", "y_plot", "fold", "numInputs"]:
        if col in df_pliegues.columns:
            df_pliegues[col] = pd.to_numeric(df_pliegues[col], errors="coerce")

    mapa_colores = {
        "X": "#1A3A5C",
        "y": "#64B5CD",
        "Xcv": "#D94F8C",
        "ycv": "#B23A48",
    }

    nombres = {
        "X": "X entrenamiento",
        "y": "y entrenamiento",
        "Xcv": "X validación",
        "ycv": "y validación",
    }

    orden = ["X", "y", "Xcv", "ycv"]

    for tipo in orden:
        df_tipo = df_pliegues[df_pliegues["tipo_segmento"] == tipo].copy()
        if df_tipo.empty:
            continue

        xs = []
        ys = []
        textos = []

        for _, fila in df_tipo.iterrows():
            xs.extend([fila["x_inicio"], fila["x_fin"], None])
            ys.extend([fila["y_plot"], fila["y_plot"], None])
            texto = (
                f"Fold: {int(fila['fold'])}<br>"
                f"Ventana: {int(fila['numInputs'])} días<br>"
                f"Segmento: {nombres.get(tipo, tipo)}<br>"
                f"Índice inicio: {int(fila['x_inicio'])}<br>"
                f"Índice fin: {int(fila['x_fin'])}"
            )
            textos.extend([texto, texto, None])

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=nombres.get(tipo, tipo),
                line=dict(color=mapa_colores.get(tipo, AZUL), width=5),
                text=textos,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    df_ticks = (
        df_pliegues[["fold", "y_plot"]]
        .dropna()
        .assign(fold=lambda d: d["fold"].astype(int))
        .groupby("fold", as_index=False)["y_plot"]
        .mean()
        .sort_values("fold")
    )

    fig.update_layout(
        title=None,
        xaxis_title="Índice temporal de la serie",
        yaxis_title="Pliegue de validación cruzada",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
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
        margin=dict(l=90, r=40, t=70, b=60),
        height=520,
    )

    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#D9E2EF",
        zeroline=False,
        tickmode="array",
        tickvals=df_ticks["y_plot"].tolist(),
        ticktext=[f"Fold {f}" for f in df_ticks["fold"].tolist()],
    )

    return fig



def figura_heatmap_bds_residuos(df_resultados_cv):
    """Construye un heatmap de p-valores BDS sin depender de nombres rígidos de columnas."""
    fig = go.Figure()

    if df_resultados_cv is None or df_resultados_cv.empty:
        fig.add_annotation(
            text="No hay datos disponibles para construir el heatmap BDS.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=360,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    df = df_resultados_cv.copy()
    df.columns = df.columns.astype(str).str.strip()

    if "ventana" in df.columns:
        y = df["ventana"].astype(str).tolist()
        y = [f"Ventana {v}" if "ventana" not in v.lower() else v for v in y]
    elif "numInputs" in df.columns:
        y = df["numInputs"].astype(str).tolist()
        y = [f"{v} días" for v in y]
    else:
        y = [f"Fila {i + 1}" for i in range(len(df))]

    columnas_fold = [
        col for col in df.columns
        if ("BDS" in col.upper())
        and ("PVALUE" in col.upper() or "P_VALUE" in col.upper() or "PVALOR" in col.upper())
        and ("FOLD" in col.upper() or "PLIEG" in col.upper())
    ]

    if columnas_fold:
        columnas_valor = columnas_fold
        x = []
        for col in columnas_valor:
            numeros = "".join([caracter for caracter in col if caracter.isdigit()])
            x.append(f"Fold {int(numeros)}" if numeros else col)
    else:
        columnas_valor = [
            col for col in ["BDS_pvalue_mean", "BDS_pvalue_min"]
            if col in df.columns
        ]

        if not columnas_valor:
            columnas_valor = [
                col for col in df.columns
                if ("BDS" in col.upper())
                and ("PVALUE" in col.upper() or "P_VALUE" in col.upper() or "PVALOR" in col.upper())
            ]

        x = []
        for col in columnas_valor:
            nombre = (
                col.replace("BDS_", "")
                .replace("pvalue", "p-valor")
                .replace("p_value", "p-valor")
                .replace("_", " ")
            )
            x.append(nombre)

    if not columnas_valor:
        fig.add_annotation(
            text="No se encontraron columnas de p-valores BDS para construir el heatmap.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=360,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    z_df = df[columnas_valor].apply(pd.to_numeric, errors="coerce")
    texto = z_df.map(lambda valor: "" if pd.isna(valor) else f"{float(valor):.3g}")

    fig = go.Figure(
        data=go.Heatmap(
            z=z_df.values,
            x=x,
            y=y,
            text=texto.values,
            texttemplate="%{text}",
            textfont=dict(size=14, color=AZUL),
            xgap=2,
            ygap=2,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.00, "#B23A48"],
                [0.05, "#F4D7D7"],
                [0.25, "#EAF1F8"],
                [1.00, AZUL],
            ],
            colorbar=dict(
                title="p-valor BDS",
                tickfont=dict(family=FUENTE, size=12, color=AZUL),
            ),
            hovertemplate=(
                "<b>Ventana:</b> %{y}<br>"
                "<b>Evaluación:</b> %{x}<br>"
                "<b>p-valor BDS:</b> %{z:.4g}<br>"
                "<b>Criterio:</b> p ≥ 0.05 no rechaza H0"
                "<extra></extra>"
            ),
        )
    )

    # Tamaño compacto para evitar que el heatmap se vea demasiado alargado
    # cuando solo existen columnas resumen como p-valor mean y p-valor min.
    n_filas = max(1, len(y))
    n_columnas = max(1, len(x))
    ancho_figura = 950
    alto_figura = 360

    fig.update_layout(
        title=None,
        xaxis_title="Fold o resumen BDS",
        yaxis_title="Ventana evaluada",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=120, r=100, t=35, b=80),
        width=ancho_figura,
        height=alto_figura,
        autosize=False,
    )

    fig.update_xaxes(
        tickangle=-25,
        showgrid=False,
        tickfont=dict(family=FUENTE, size=13, color=AZUL),
        linecolor="#D9E2EF",
        linewidth=1,
        mirror=True,
    )
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        tickfont=dict(family=FUENTE, size=13, color=AZUL),
        linecolor="#D9E2EF",
        linewidth=1,
        mirror=True,
    )

    return fig

def layout_svr_calamar():
    metadata, df_test = cargar_resultados_svr_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes = cargar_tablas_svr_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_svr(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_svr(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación interna", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo final", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata.get("best_params", {})
    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "SVR multioutput")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "kernel", "Valor": best_params.get("modelo__estimator__kernel", best_params.get("svr__kernel", "N/A"))},
        {"Parámetro": "C", "Valor": best_params.get("modelo__estimator__C", best_params.get("svr__C", "N/A"))},
        {"Parámetro": "epsilon", "Valor": best_params.get("modelo__estimator__epsilon", best_params.get("svr__epsilon", "N/A"))},
        {"Parámetro": "gamma", "Valor": best_params.get("modelo__estimator__gamma", best_params.get("svr__gamma", "N/A"))},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(df_metricas_horizontes_tabla[col], errors="coerce").round(4)

    # Tabla específica para el test BDS de los residuos durante la validación cruzada.
    # H0 del BDS: los residuos son i.i.d. Si p-valor >= 0.05, no se rechaza H0.
    columnas_bds = [
        "ventana",
        "numInputs",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Máquina de Vectores de Soporte (SVR) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y MultiOutputRegressor(SVR), "
                "configurado para predecir simultáneamente un horizonte de 10 días del nivel en la estación Calamar. "
                "La selección se realizó con validación cruzada temporal mediante split_train_val_groupKFold.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada e hiperparámetros del SVR dentro de un esquema multioutput. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=12),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo SVR multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_residuos_svr_multioutput_h10",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_svr_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo SVR multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


def layout_modelos(modelos=None):
    fig_pliegues = figura_pliegues_validacion()

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Modelos implementados", style=estilo_titulo),
            html.P(
                "Debido a la alta multicolinealidad observada entre las series de nivel de las diferentes estaciones, se decidió "
                "construir los modelos utilizando únicamente la información temporal y el nivel registrado en Calamar. Esta "
                "decisión se tomó especialmente para evitar problemas en los modelos lineales, como Ridge y Lasso, en los "
                "que la presencia de variables altamente correlacionadas puede afectar la estabilidad e interpretación de los "
                "coeficientes. De esta forma, se trabajó bajo un esquema univariado, donde el nivel futuro se predice a partir de "
                "la memoria reciente de la propia serie.",
                style=estilo_parrafo,
            ),
            html.P(
                "La gráfica muestra los segmentos de entrada y salida utilizados en los pliegues de validación cruzada temporalpara todos los modelos. ",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_pliegues,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "pliegues_validacion_cruzada_temporal_svr_multioutput_h10",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.Br(),
            html.P(
                "A continuación seleccione un modelo para visualizar su configuración, validación temporal, métricas, predicciones y diagnóstico de residuos "
                "en la estación de Calamar.",
                style=estilo_parrafo,
            ),
            dcc.Dropdown(
                id="selector-modelo",
                options=[
                    {"label": "Todos", "value": "todos"},
                    {"label": "Máquina de Vectores de Soporte (SVR)", "value": "svr_calamar"},
                    {"label": "K-Vecinos más Cercanos (KNN)", "value": "knn_calamar"},
                    {"label": "XARIMA/ARIMA", "value": "xarima_calamar"},
                    {"label": "Árbol de Decisión", "value": "dt_calamar"},
                    {"label": "Regresión Lasso", "value": "lasso_calamar"},
                    {"label": "Regresión Ridge", "value": "ridge_calamar"},
                    {"label": "Random Forest", "value": "rf_calamar"},
                    {"label": "XGBoost", "value": "xgb_calamar"},
                    {"label": "Multi-Layer Perceptron (MLP)", "value": "mlp_calamar"},
                    {"label": "Recurrent Neural Network (RNN)", "value": "rnn_calamar"},
                    {"label": "Long Short-Term Memory (LSTM)", "value": "lstm_calamar"},
                    {"label": "Convolutional Neural Network (CNN)", "value": "cnn_calamar"},
                ],
                value="svr_calamar",
                clearable=False,
                style={"fontFamily": FUENTE, "fontSize": "14px", "maxWidth": "520px"},
            ),
        ]),
        html.Div(id="contenido-modelo"),
    ])


# =====================================================================
# ACTUALIZACIÓN RIDGE MULTIOUTPUT H10
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para Ridge multioutput H10
RUTA_METADATA_RIDGE = _resolver_archivo_resultados(
    "metadata_modelo_ridge_multioutput_h10_calamar.json",
    subcarpetas=("Ridge", "2_Ridge"),
)
RUTA_TEST_RIDGE = _resolver_archivo_resultados(
    "test_final_externo_ridge_multioutput_h10_calamar.csv",
    subcarpetas=("Ridge", "2_Ridge"),
)
RUTA_MODELO_RIDGE = _resolver_archivo_resultados(
    "modelo_ridge_multioutput_h10_calamar.joblib",
    subcarpetas=("Ridge", "2_Ridge"),
)
RUTA_RESUMEN_RIDGE = _resolver_archivo_resultados(
    "resumen_ridge_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("Ridge", "2_Ridge"),
)
RUTA_RESULTADOS_CV_RIDGE = _resolver_archivo_resultados(
    "resultados_ridge_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("Ridge", "2_Ridge"),
)
RUTA_METRICAS_HORIZONTES_RIDGE = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_ridge_multioutput_h10.csv",
    subcarpetas=("Ridge", "2_Ridge"),
)

# Actualizar la entrada Ridge de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "ridge_calamar":
        _spec["nombre"] = "Ridge Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_RIDGE
        _spec["ruta_test"] = RUTA_TEST_RIDGE


def cargar_resultados_ridge_calamar():
    with open(RUTA_METADATA_RIDGE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_modelos(RUTA_TEST_RIDGE)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_ridge_multioutput():
    df_resumen = _leer_csv_modelos(RUTA_RESUMEN_RIDGE)
    df_resultados_cv = _leer_csv_modelos(RUTA_RESULTADOS_CV_RIDGE)
    df_metricas_horizontes = _leer_csv_modelos(RUTA_METRICAS_HORIZONTES_RIDGE)
    return df_resumen, df_resultados_cv, df_metricas_horizontes


def layout_ridge_calamar():
    metadata, df_test = cargar_resultados_ridge_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes = cargar_tablas_ridge_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_ridge(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_ridge(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)

    df_validacion = pd.DataFrame([
        {
            "Conjunto": "Train / validación interna",
            "Fecha inicial": metadata["fecha_inicio_trainval"],
            "Fecha final": metadata["fecha_fin_trainval"],
        },
        {
            "Conjunto": "Test externo final",
            "Fecha inicial": metadata["fecha_inicio_test_externo"],
            "Fecha final": metadata["fecha_fin_test_externo"],
        },
    ])

    best_params = metadata.get("best_params", {})
    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "Pipeline(StandardScaler + Ridge)")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "alpha", "Valor": best_params.get("ridge__alpha", best_params.get("alpha", "N/A"))},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {
            "Etapa": "Test externo H10",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
        }
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(
                df_metricas_horizontes_tabla[col],
                errors="coerce"
            ).round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "alpha",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "alpha": "Alpha",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Regresión Ridge - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y Ridge, "
                "configurado para predecir simultáneamente un horizonte de 10 días del nivel en la estación Calamar. "
                "La selección se realizó con validación cruzada temporal mediante split_train_val_groupKFold.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada y valores del parámetro alpha dentro de un esquema multioutput. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=12),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada y valor de alpha evaluado "
                "en el modelo Ridge multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_ridge_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_ridge_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo Ridge multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])

# =====================================================================
# ACTUALIZACIÓN LASSO MULTIOUTPUT H10
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para Lasso multioutput H10
RUTA_METADATA_LASSO = _resolver_archivo_resultados(
    "metadata_modelo_lasso_multioutput_h10_calamar.json",
    subcarpetas=("Lasso", "3_Lasso"),
)
RUTA_TEST_LASSO = _resolver_archivo_resultados(
    "test_final_externo_lasso_multioutput_h10_calamar.csv",
    subcarpetas=("Lasso", "3_Lasso"),
)
RUTA_MODELO_LASSO = _resolver_archivo_resultados(
    "modelo_lasso_multioutput_h10_calamar.joblib",
    subcarpetas=("Lasso", "3_Lasso"),
)
RUTA_RESUMEN_LASSO = _resolver_archivo_resultados(
    "resumen_lasso_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("Lasso", "3_Lasso"),
)
RUTA_RESULTADOS_CV_LASSO = _resolver_archivo_resultados(
    "resultados_lasso_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("Lasso", "3_Lasso"),
)
RUTA_METRICAS_HORIZONTES_LASSO = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_lasso_multioutput_h10.csv",
    subcarpetas=("Lasso", "3_Lasso"),
)

# Actualizar la entrada Lasso de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "lasso_calamar":
        _spec["nombre"] = "Lasso Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_LASSO
        _spec["ruta_test"] = RUTA_TEST_LASSO


def _limpiar_columnas_bom_lasso(df):
    """Limpia BOM en columnas de los archivos de Lasso sin cambiar la función general de lectura."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def cargar_resultados_lasso_calamar():
    with open(RUTA_METADATA_LASSO, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _limpiar_columnas_bom_lasso(_leer_csv_modelos(RUTA_TEST_LASSO))

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_lasso_multioutput():
    df_resumen = _limpiar_columnas_bom_lasso(_leer_csv_modelos(RUTA_RESUMEN_LASSO))
    df_resultados_cv = _limpiar_columnas_bom_lasso(_leer_csv_modelos(RUTA_RESULTADOS_CV_LASSO))
    df_metricas_horizontes = _limpiar_columnas_bom_lasso(_leer_csv_modelos(RUTA_METRICAS_HORIZONTES_LASSO))
    return df_resumen, df_resultados_cv, df_metricas_horizontes


def layout_lasso_calamar():
    metadata, df_test = cargar_resultados_lasso_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes = cargar_tablas_lasso_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_lasso(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_lasso(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)

    df_validacion = pd.DataFrame([
        {
            "Conjunto": "Train / validación interna",
            "Fecha inicial": metadata["fecha_inicio_trainval"],
            "Fecha final": metadata["fecha_fin_trainval"],
        },
        {
            "Conjunto": "Test externo final",
            "Fecha inicial": metadata["fecha_inicio_test_externo"],
            "Fecha final": metadata["fecha_fin_test_externo"],
        },
    ])

    best_params = metadata.get("best_params", {})
    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "Pipeline(StandardScaler + Lasso)")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "alpha", "Valor": best_params.get("lasso__alpha", best_params.get("alpha", "N/A"))},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {
            "Etapa": "Test externo H10",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
        }
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(
                df_metricas_horizontes_tabla[col],
                errors="coerce"
            ).round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "alpha",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "alpha": "Alpha",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Regresión Lasso - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y Lasso, "
                "configurado para predecir simultáneamente un horizonte de 10 días del nivel en la estación Calamar. "
                "La selección se realizó con validación cruzada temporal mediante split_train_val_groupKFold.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada y valores del parámetro alpha dentro de un esquema multioutput. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=12),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada y valor de alpha evaluado "
                "en el modelo Lasso multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_lasso_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_lasso_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo Lasso multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])

# =====================================================================
# ACTUALIZACIÓN DECISION TREE MULTIOUTPUT H10
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================


def _resolver_archivo_dt_resultados(nombre_archivo, subcarpetas=("DecisionTree", "4_DecisionTree", "DT", "Decision_Tree")):
    """Busca archivos del modelo Decision Tree en las rutas más probables."""
    candidatos = []
    for sub in subcarpetas:
        candidatos.append(Path("Resultados") / sub / nombre_archivo)
    candidatos.append(Path("Resultados") / nombre_archivo)
    for sub in subcarpetas:
        candidatos.append(Path(sub) / nombre_archivo)

    for ruta in candidatos:
        if ruta.exists():
            return ruta
    return candidatos[0]


# Nuevas rutas para Decision Tree multioutput H10
RUTA_METADATA_DT = _resolver_archivo_dt_resultados(
    "metadata_modelo_dt_multioutput_h10_calamar.json"
)
RUTA_TEST_DT = _resolver_archivo_dt_resultados(
    "test_final_externo_dt_multioutput_h10_calamar.csv"
)
RUTA_MODELO_DT = _resolver_archivo_dt_resultados(
    "modelo_dt_multioutput_h10_calamar.joblib"
)
RUTA_RESUMEN_DT = _resolver_archivo_dt_resultados(
    "resumen_dt_multioutput_h10_timeseries_cv_bds.csv"
)
RUTA_RESULTADOS_CV_DT = _resolver_archivo_dt_resultados(
    "resultados_dt_multioutput_h10_timeseries_cv_bds.csv"
)
RUTA_METRICAS_HORIZONTES_DT = _resolver_archivo_dt_resultados(
    "metricas_horizontes_1_5_10_dt_multioutput_h10.csv"
)

# Actualizar la entrada Decision Tree de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "dt_calamar":
        _spec["nombre"] = "Decision Tree Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_DT
        _spec["ruta_test"] = RUTA_TEST_DT


def _leer_csv_dt_modelos(ruta):
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def cargar_resultados_dt_calamar():
    with open(RUTA_METADATA_DT, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_dt_modelos(RUTA_TEST_DT)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_dt_multioutput():
    df_resumen = _leer_csv_dt_modelos(RUTA_RESUMEN_DT)
    df_resultados_cv = _leer_csv_dt_modelos(RUTA_RESULTADOS_CV_DT)
    df_metricas_horizontes = _leer_csv_dt_modelos(RUTA_METRICAS_HORIZONTES_DT)
    return df_resumen, df_resultados_cv, df_metricas_horizontes


def figura_heatmap_bds_dt_residuos(df_resultados_cv):
    """Construye un heatmap de p-valores BDS para Decision Tree."""
    fig = go.Figure()

    if df_resultados_cv is None or df_resultados_cv.empty:
        fig.add_annotation(
            text="No hay datos disponibles para construir el heatmap BDS.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=360,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    df = df_resultados_cv.copy()
    df.columns = df.columns.astype(str).str.strip()

    if "ventana" in df.columns:
        y = df["ventana"].astype(str).tolist()
        y = [f"Ventana {v}" if "ventana" not in v.lower() else v for v in y]
    elif "numInputs" in df.columns:
        y = df["numInputs"].astype(str).tolist()
        y = [f"{v} días" for v in y]
    else:
        y = [f"Fila {i + 1}" for i in range(len(df))]

    columnas_valor = [
        col for col in ["BDS_pvalue_mean", "BDS_pvalue_min"]
        if col in df.columns
    ]

    if not columnas_valor:
        columnas_valor = [
            col for col in df.columns
            if ("BDS" in col.upper())
            and ("PVALUE" in col.upper() or "P_VALUE" in col.upper() or "PVALOR" in col.upper())
        ]

    if not columnas_valor:
        fig.add_annotation(
            text="No se encontraron columnas de p-valores BDS para construir el heatmap.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=360,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    x = []
    for col in columnas_valor:
        nombre = (
            col.replace("BDS_", "")
            .replace("pvalue", "p-valor")
            .replace("p_value", "p-valor")
            .replace("_", " ")
        )
        x.append(nombre)

    z_df = df[columnas_valor].apply(pd.to_numeric, errors="coerce")
    texto = z_df.map(lambda valor: "" if pd.isna(valor) else f"{float(valor):.3g}")

    fig = go.Figure(
        data=go.Heatmap(
            z=z_df.values,
            x=x,
            y=y,
            text=texto.values,
            texttemplate="%{text}",
            textfont=dict(size=14, color=AZUL),
            xgap=2,
            ygap=2,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.00, "#B23A48"],
                [0.05, "#F4D7D7"],
                [0.25, "#EAF1F8"],
                [1.00, AZUL],
            ],
            colorbar=dict(
                title="p-valor BDS",
                tickfont=dict(family=FUENTE, size=12, color=AZUL),
            ),
            hovertemplate=(
                "<b>Ventana:</b> %{y}<br>"
                "<b>Evaluación:</b> %{x}<br>"
                "<b>p-valor BDS:</b> %{z:.4g}<br>"
                "<b>Criterio:</b> p ≥ 0.05 no rechaza H0"
                "<extra></extra>"
            ),
        )
    )

    n_filas = max(1, len(y))
    n_columnas = max(1, len(x))
    ancho_figura = 950
    alto_figura = 360

    fig.update_layout(
        title=None,
        xaxis_title="Fold o resumen BDS",
        yaxis_title="Ventana evaluada",
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family=FUENTE, size=13, color=AZUL),
        margin=dict(l=120, r=100, t=35, b=80),
        width=ancho_figura,
        height=alto_figura,
        autosize=False,
    )

    fig.update_xaxes(
        tickangle=-25,
        showgrid=False,
        tickfont=dict(family=FUENTE, size=13, color=AZUL),
        linecolor="#D9E2EF",
        linewidth=1,
        mirror=True,
    )
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        tickfont=dict(family=FUENTE, size=13, color=AZUL),
        linecolor="#D9E2EF",
        linewidth=1,
        mirror=True,
    )

    return fig


def layout_dt_calamar():
    metadata, df_test = cargar_resultados_dt_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes = cargar_tablas_dt_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_dt(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_dt(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_dt_residuos(df_resultados_cv)

    df_validacion = pd.DataFrame([
        {
            "Conjunto": "Train / validación interna",
            "Fecha inicial": metadata["fecha_inicio_trainval"],
            "Fecha final": metadata["fecha_fin_trainval"],
        },
        {
            "Conjunto": "Test externo final",
            "Fecha inicial": metadata["fecha_inicio_test_externo"],
            "Fecha final": metadata["fecha_fin_test_externo"],
        },
    ])

    best_params = metadata.get("best_params", {})

    def _valor_parametro_dt(valor):
        if valor is None or pd.isna(valor):
            return "None"
        return valor

    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "DecisionTreeRegressor multioutput nativo")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "random_state", "Valor": best_params.get("random_state", "N/A")},
        {"Parámetro": "max_depth", "Valor": _valor_parametro_dt(best_params.get("max_depth", "N/A"))},
        {"Parámetro": "min_samples_split", "Valor": best_params.get("min_samples_split", "N/A")},
        {"Parámetro": "min_samples_leaf", "Valor": best_params.get("min_samples_leaf", "N/A")},
        {"Parámetro": "max_features", "Valor": _valor_parametro_dt(best_params.get("max_features", "N/A"))},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {
            "Etapa": "Test externo H10",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
        }
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(
                df_metricas_horizontes_tabla[col],
                errors="coerce"
            ).round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "random_state",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "random_state": "Random state",
        "max_depth": "Max depth",
        "min_samples_split": "Min samples split",
        "min_samples_leaf": "Min samples leaf",
        "max_features": "Max features",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds_dt(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds_dt(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds_dt(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Árbol de Decisión - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un DecisionTreeRegressor multioutput nativo, "
                "configurado para predecir simultáneamente un horizonte de 10 días del nivel en la estación Calamar. "
                "La selección se realizó con validación cruzada temporal mediante split_train_val_groupKFold.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada e hiperparámetros del árbol de decisión dentro de un esquema multioutput. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=12),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo Decision Tree multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_dt_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_dt_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo Decision Tree multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


# =====================================================================
# ACTUALIZACIÓN RANDOM FOREST MULTIOUTPUT H10
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================


# Nuevas rutas para Random Forest multioutput H10
RUTA_METADATA_RF = _resolver_archivo_resultados(
    "metadata_modelo_rf_multioutput_h10_calamar.json",
    subcarpetas=("RandomForest", "5_RandomForest", "RF", "Random_Forest"),
)
RUTA_TEST_RF = _resolver_archivo_resultados(
    "test_final_externo_rf_multioutput_h10_calamar.csv",
    subcarpetas=("RandomForest", "5_RandomForest", "RF", "Random_Forest"),
)
RUTA_MODELO_RF = _resolver_archivo_resultados(
    "modelo_rf_multioutput_h10_calamar.joblib",
    subcarpetas=("RandomForest", "5_RandomForest", "RF", "Random_Forest"),
)
RUTA_RESUMEN_RF = _resolver_archivo_resultados(
    "resumen_rf_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("RandomForest", "5_RandomForest", "RF", "Random_Forest"),
)
RUTA_RESULTADOS_CV_RF = _resolver_archivo_resultados(
    "resultados_rf_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("RandomForest", "5_RandomForest", "RF", "Random_Forest"),
)
RUTA_METRICAS_HORIZONTES_RF = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_rf_multioutput_h10.csv",
    subcarpetas=("RandomForest", "5_RandomForest", "RF", "Random_Forest"),
)

# Actualizar la entrada Random Forest de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "rf_calamar":
        _spec["nombre"] = "Random Forest Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_RF
        _spec["ruta_test"] = RUTA_TEST_RF


def _leer_csv_rf_modelos(ruta):
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def cargar_resultados_rf_calamar():
    with open(RUTA_METADATA_RF, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_rf_modelos(RUTA_TEST_RF)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_rf_multioutput():
    df_resumen = _leer_csv_rf_modelos(RUTA_RESUMEN_RF)
    df_resultados_cv = _leer_csv_rf_modelos(RUTA_RESULTADOS_CV_RF)
    df_metricas_horizontes = _leer_csv_rf_modelos(RUTA_METRICAS_HORIZONTES_RF)
    return df_resumen, df_resultados_cv, df_metricas_horizontes


def layout_rf_calamar():
    metadata, df_test = cargar_resultados_rf_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes = cargar_tablas_rf_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_rf(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_rf(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)

    df_validacion = pd.DataFrame([
        {
            "Conjunto": "Train / validación interna",
            "Fecha inicial": metadata["fecha_inicio_trainval"],
            "Fecha final": metadata["fecha_fin_trainval"],
        },
        {
            "Conjunto": "Test externo final",
            "Fecha inicial": metadata["fecha_inicio_test_externo"],
            "Fecha final": metadata["fecha_fin_test_externo"],
        },
    ])

    best_params = metadata.get("best_params", {})

    def _valor_parametro_rf(valor):
        return "None" if valor is None or pd.isna(valor) else valor

    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "RandomForestRegressor multioutput")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "n_estimators", "Valor": best_params.get("n_estimators", "N/A")},
        {"Parámetro": "max_depth", "Valor": _valor_parametro_rf(best_params.get("max_depth", "N/A"))},
        {"Parámetro": "min_samples_leaf", "Valor": best_params.get("min_samples_leaf", "N/A")},
        {"Parámetro": "max_features", "Valor": best_params.get("max_features", "N/A")},
        {"Parámetro": "random_state", "Valor": best_params.get("random_state", "N/A")},
        {"Parámetro": "n_jobs", "Valor": best_params.get("n_jobs", "N/A")},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(
                df_metricas_horizontes_tabla[col], errors="coerce"
            ).round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds_rf(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds_rf(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds_rf(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Random Forest - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un RandomForestRegressor multioutput nativo, configurado para predecir "
                "simultáneamente un horizonte de 10 días del nivel en la estación Calamar. La selección se realizó "
                "con validación cruzada temporal mediante split_train_val_groupKFold.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada e hiperparámetros del Random Forest dentro de un esquema multioutput. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=12),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo Random Forest multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_rf_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_rf_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo Random Forest multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


# =====================================================================
# ACTUALIZACIÓN XGBOOST MULTIOUTPUT H10
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para XGBoost multioutput H10
RUTA_METADATA_XGB = _resolver_archivo_resultados(
    "metadata_modelo_xgb_multioutput_h10_calamar.json",
    subcarpetas=("XGBoost", "6_XGBoost", "XGB", "XGBoost_MultiOutput"),
)
RUTA_TEST_XGB = _resolver_archivo_resultados(
    "test_final_externo_xgb_multioutput_h10_calamar.csv",
    subcarpetas=("XGBoost", "6_XGBoost", "XGB", "XGBoost_MultiOutput"),
)
RUTA_MODELO_XGB = _resolver_archivo_resultados(
    "modelo_xgb_multioutput_h10_calamar.joblib",
    subcarpetas=("XGBoost", "6_XGBoost", "XGB", "XGBoost_MultiOutput"),
)
RUTA_RESUMEN_XGB = _resolver_archivo_resultados(
    "resumen_xgb_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("XGBoost", "6_XGBoost", "XGB", "XGBoost_MultiOutput"),
)
RUTA_RESULTADOS_CV_XGB = _resolver_archivo_resultados(
    "resultados_xgb_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("XGBoost", "6_XGBoost", "XGB", "XGBoost_MultiOutput"),
)
RUTA_METRICAS_HORIZONTES_XGB = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_xgb_multioutput_h10.csv",
    subcarpetas=("XGBoost", "6_XGBoost", "XGB", "XGBoost_MultiOutput"),
)

# Actualizar la entrada XGBoost de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "xgb_calamar":
        _spec["nombre"] = "XGBoost Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_XGB
        _spec["ruta_test"] = RUTA_TEST_XGB


def _leer_csv_xgb_modelos(ruta):
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def cargar_resultados_xgb_calamar():
    with open(RUTA_METADATA_XGB, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_xgb_modelos(RUTA_TEST_XGB)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_xgb_multioutput():
    df_resumen = _leer_csv_xgb_modelos(RUTA_RESUMEN_XGB)
    df_resultados_cv = _leer_csv_xgb_modelos(RUTA_RESULTADOS_CV_XGB)
    df_metricas_horizontes = _leer_csv_xgb_modelos(RUTA_METRICAS_HORIZONTES_XGB)
    return df_resumen, df_resultados_cv, df_metricas_horizontes


def layout_xgb_calamar():
    metadata, df_test = cargar_resultados_xgb_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes = cargar_tablas_xgb_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_xgb(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_xgb(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)

    df_validacion = pd.DataFrame([
        {
            "Conjunto": "Train / validación interna",
            "Fecha inicial": metadata["fecha_inicio_trainval"],
            "Fecha final": metadata["fecha_fin_trainval"],
        },
        {
            "Conjunto": "Test externo final",
            "Fecha inicial": metadata["fecha_inicio_test_externo"],
            "Fecha final": metadata["fecha_fin_test_externo"],
        },
    ])

    best_params = metadata.get("best_params", {})

    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "MultiOutputRegressor(XGBRegressor)")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "objective", "Valor": best_params.get("objective", "N/A")},
        {"Parámetro": "n_estimators", "Valor": best_params.get("n_estimators", "N/A")},
        {"Parámetro": "max_depth", "Valor": best_params.get("max_depth", "N/A")},
        {"Parámetro": "learning_rate", "Valor": best_params.get("learning_rate", "N/A")},
        {"Parámetro": "subsample", "Valor": best_params.get("subsample", "N/A")},
        {"Parámetro": "colsample_bytree", "Valor": best_params.get("colsample_bytree", "N/A")},
        {"Parámetro": "reg_lambda", "Valor": best_params.get("reg_lambda", "N/A")},
        {"Parámetro": "random_state", "Valor": best_params.get("random_state", "N/A")},
        {"Parámetro": "n_jobs", "Valor": best_params.get("n_jobs", "N/A")},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(
                df_metricas_horizontes_tabla[col], errors="coerce"
            ).round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds_xgb(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds_xgb(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds_xgb(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("XGBoost - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un MultiOutputRegressor(XGBRegressor), configurado para predecir "
                "simultáneamente un horizonte de 10 días del nivel en la estación Calamar. La selección se realizó "
                "con validación cruzada temporal mediante split_train_val_groupKFold.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada e hiperparámetros del XGBoost dentro de un esquema multioutput. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=15),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo XGBoost multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_xgb_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_xgb_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo XGBoost multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


# =====================================================================
# ACTUALIZACIÓN LSTM MULTIOUTPUT H10 + CURVA DE APRENDIZAJE
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para LSTM multioutput H10
RUTA_METADATA_LSTM = _resolver_archivo_resultados(
    "metadata_modelo_lstm_multioutput_h10_calamar.json",
    subcarpetas=("LSTM", "7_LSTM", "ResultadosLSTM_MultiOutput_H10"),
)
RUTA_TEST_LSTM = _resolver_archivo_resultados(
    "test_final_externo_lstm_multioutput_h10_calamar.csv",
    subcarpetas=("LSTM", "7_LSTM", "ResultadosLSTM_MultiOutput_H10"),
)
RUTA_MODELO_LSTM = _resolver_archivo_resultados(
    "modelo_lstm_multioutput_h10_calamar.keras",
    subcarpetas=("LSTM", "7_LSTM", "ResultadosLSTM_MultiOutput_H10"),
)
RUTA_RESUMEN_LSTM = _resolver_archivo_resultados(
    "resumen_lstm_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("LSTM", "7_LSTM", "ResultadosLSTM_MultiOutput_H10"),
)
RUTA_RESULTADOS_CV_LSTM = _resolver_archivo_resultados(
    "resultados_lstm_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("LSTM", "7_LSTM", "ResultadosLSTM_MultiOutput_H10"),
)
RUTA_METRICAS_HORIZONTES_LSTM = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_lstm_multioutput_h10.csv",
    subcarpetas=("LSTM", "7_LSTM", "ResultadosLSTM_MultiOutput_H10"),
)
RUTA_HISTORIAL_ENTRENAMIENTO_LSTM = _resolver_archivo_resultados(
    "historial_entrenamiento_lstm_multioutput_h10.csv",
    subcarpetas=("LSTM", "7_LSTM", "ResultadosLSTM_MultiOutput_H10"),
)
RUTA_HISTORIAL_MODELO_FINAL_LSTM = _resolver_archivo_resultados(
    "historial_modelo_final_lstm_multioutput_h10.csv",
    subcarpetas=("LSTM", "7_LSTM", "ResultadosLSTM_MultiOutput_H10"),
)

# Actualizar la entrada LSTM de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "lstm_calamar":
        _spec["nombre"] = "LSTM Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_LSTM
        _spec["ruta_test"] = RUTA_TEST_LSTM


def cargar_resultados_lstm_calamar():
    with open(RUTA_METADATA_LSTM, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_modelos(RUTA_TEST_LSTM)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_lstm_multioutput():
    df_resumen = _leer_csv_modelos(RUTA_RESUMEN_LSTM)
    df_resultados_cv = _leer_csv_modelos(RUTA_RESULTADOS_CV_LSTM)
    df_metricas_horizontes = _leer_csv_modelos(RUTA_METRICAS_HORIZONTES_LSTM)
    df_historial_entrenamiento = _leer_csv_modelos(RUTA_HISTORIAL_ENTRENAMIENTO_LSTM)
    df_historial_final = _leer_csv_modelos(RUTA_HISTORIAL_MODELO_FINAL_LSTM)
    return df_resumen, df_resultados_cv, df_metricas_horizontes, df_historial_entrenamiento, df_historial_final


def figura_curva_aprendizaje_lstm(df_historial_entrenamiento, df_historial_final=None, num_inputs=None):
    """Curva de aprendizaje del LSTM usando el historial de entrenamiento."""
    fig = go.Figure()

    if df_historial_entrenamiento is None or df_historial_entrenamiento.empty:
        fig.add_annotation(
            text="No hay datos disponibles para construir la curva de aprendizaje.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=420,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    df = df_historial_entrenamiento.copy()
    df.columns = df.columns.astype(str).str.strip()

    for col in ["epoch", "loss_train", "loss_val", "mae_train", "mae_val", "numInputs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if num_inputs is not None and "numInputs" in df.columns:
        df_filtrado = df[df["numInputs"] == float(num_inputs)].copy()
        if not df_filtrado.empty:
            df = df_filtrado

    columnas_agrupar = ["epoch"]
    columnas_media = [col for col in ["loss_train", "loss_val", "mae_train", "mae_val"] if col in df.columns]
    df_prom = df.groupby(columnas_agrupar, as_index=False)[columnas_media].mean().sort_values("epoch")

    if "loss_train" in df_prom.columns:
        fig.add_trace(go.Scatter(
            x=df_prom["epoch"],
            y=df_prom["loss_train"],
            mode="lines",
            name="Train loss CV",
            line=dict(color=AZUL, width=2.5),
            hovertemplate="<b>Época:</b> %{x}<br><b>Loss train:</b> %{y:.5f}<extra></extra>",
        ))

    if "loss_val" in df_prom.columns:
        fig.add_trace(go.Scatter(
            x=df_prom["epoch"],
            y=df_prom["loss_val"],
            mode="lines",
            name="Validation loss CV",
            line=dict(color=CELESTE, width=2.5, dash="dash"),
            hovertemplate="<b>Época:</b> %{x}<br><b>Loss validación:</b> %{y:.5f}<extra></extra>",
        ))

    epoca_final_usada = None
    loss_epoca_final = None

    if df_historial_final is not None and not df_historial_final.empty:
        df_final = df_historial_final.copy()
        df_final.columns = df_final.columns.astype(str).str.strip()
        for col in ["epoch", "loss_train", "loss_val", "epochs_usados"]:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

        if "loss_train" in df_final.columns:
            fig.add_trace(go.Scatter(
                x=df_final["epoch"],
                y=df_final["loss_train"],
                mode="lines",
                name="Train loss modelo final",
                line=dict(color="#B23A48", width=2.2, dash="dot"),
                hovertemplate="<b>Época:</b> %{x}<br><b>Loss modelo final:</b> %{y:.5f}<extra></extra>",
            ))

        if "epochs_usados" in df_final.columns and df_final["epochs_usados"].notna().any():
            epoca_final_usada = int(df_final["epochs_usados"].dropna().max())
        elif "epoch" in df_final.columns and df_final["epoch"].notna().any():
            epoca_final_usada = int(df_final["epoch"].dropna().max())

        if epoca_final_usada is not None and {"epoch", "loss_train"}.issubset(df_final.columns):
            df_epoca_final = df_final[df_final["epoch"] == epoca_final_usada].copy()
            if not df_epoca_final.empty:
                loss_epoca_final = float(df_epoca_final["loss_train"].iloc[-1])
                fig.add_trace(go.Scatter(
                    x=[epoca_final_usada],
                    y=[loss_epoca_final],
                    mode="markers",
                    name=f"Época final usada ({epoca_final_usada})",
                    marker=dict(size=12, color="#B23A48", symbol="diamond", line=dict(width=2, color=AZUL)),
                    hovertemplate=(
                        "<b>Época final usada:</b> %{x}<br>"
                        "<b>Loss final:</b> %{y:.5f}"
                        "<extra></extra>"
                    ),
                ))

    if epoca_final_usada is not None:
        fig.add_vline(
            x=epoca_final_usada,
            line_dash="dash",
            line_width=2,
            line_color="#B23A48",
            annotation_text=f"Época final usada: {epoca_final_usada}",
            annotation_position="top right",
            annotation_font=dict(family=FUENTE, size=12, color="#B23A48"),
        )

    # Señalar explícitamente la época con menor loss de validación (val_loss)
    # usando el menor valor real del historial de validación, no el promedio por época.
    epoca_menor_val_loss = None
    valor_menor_val_loss = None
    fold_menor_val_loss = None

    if {"epoch", "loss_val"}.issubset(df.columns) and df["loss_val"].notna().any():
        idx_min_val = df["loss_val"].idxmin()
        epoca_menor_val_loss = int(df.loc[idx_min_val, "epoch"])
        valor_menor_val_loss = float(df.loc[idx_min_val, "loss_val"])
        if "fold" in df.columns and pd.notna(df.loc[idx_min_val, "fold"]):
            fold_menor_val_loss = int(df.loc[idx_min_val, "fold"])

        texto_menor_val = f"Menor val loss: época {epoca_menor_val_loss}"
        if fold_menor_val_loss is not None:
            texto_menor_val += f" | fold {fold_menor_val_loss}"

        fig.add_vline(
            x=epoca_menor_val_loss,
            line_dash="dash",
            line_width=2.5,
            line_color="#8E44AD",
            annotation_text=texto_menor_val,
            annotation_position="top left",
            annotation_font=dict(family=FUENTE, size=12, color="#8E44AD"),
        )

        fig.add_trace(go.Scatter(
            x=[epoca_menor_val_loss],
            y=[valor_menor_val_loss],
            mode="markers+text",
            name=f"Menor val loss ({epoca_menor_val_loss})",
            text=[f"mín val loss<br>época {epoca_menor_val_loss}"],
            textposition="bottom right",
            marker=dict(
                size=14,
                color="#8E44AD",
                symbol="diamond",
                line=dict(width=2, color=AZUL),
            ),
            hovertemplate=(
                "<b>Menor val loss</b><br>"
                "<b>Época:</b> %{x}<br>"
                "<b>val_loss:</b> %{y:.6f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=None,
        xaxis_title="Época",
        yaxis_title="Loss",
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
        margin=dict(l=70, r=40, t=80, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)

    return fig


def layout_lstm_calamar():
    metadata, df_test = cargar_resultados_lstm_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes, df_historial_entrenamiento, df_historial_final = cargar_tablas_lstm_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_lstm(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_lstm(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)
    fig_curva_aprendizaje = figura_curva_aprendizaje_lstm(
        df_historial_entrenamiento,
        df_historial_final,
        num_inputs=metadata.get("numInputs_seleccionado", metadata.get("numInputs")),
    )

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación interna", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo final", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata.get("best_params", {})
    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "Keras LSTM MultiOutput Dense(10)")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "units", "Valor": best_params.get("units", "N/A")},
        {"Parámetro": "dropout", "Valor": best_params.get("dropout", "N/A")},
        {"Parámetro": "learning_rate", "Valor": best_params.get("learning_rate", "N/A")},
        {"Parámetro": "batch_size", "Valor": best_params.get("batch_size", "N/A")},
        {"Parámetro": "epochs_max", "Valor": best_params.get("epochs_max", "N/A")},
        {"Parámetro": "patience", "Valor": best_params.get("patience", "N/A")},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(df_metricas_horizontes_tabla[col], errors="coerce").round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Long Short-Term Memory (LSTM) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una red LSTM implementada en Keras, configurada para predecir simultáneamente "
                "un horizonte de 10 días del nivel en la estación Calamar. La selección se realizó con validación cruzada "
                "temporal mediante split_train_val_groupKFold y diagnóstico BDS de residuos.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
            html.P(metadata.get("nota_y_scaling", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada e hiperparámetros de la red LSTM. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=12),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Curva de aprendizaje", style=estilo_titulo),
            html.P(
                "La curva de aprendizaje se construyó a partir del historial de entrenamiento. "
                "Se muestran las pérdidas promedio de entrenamiento y validación durante la validación cruzada, "
                "junto con la pérdida de entrenamiento del modelo final.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_curva_aprendizaje,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "curva_aprendizaje_lstm_multioutput_h10",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo LSTM multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_lstm_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_lstm_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo LSTM multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


# =====================================================================
# ACTUALIZACIÓN KNN MULTIOUTPUT H10
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para KNN multioutput H10
RUTA_METADATA_KNN = _resolver_archivo_resultados(
    "metadata_modelo_knn_multioutput_h10_calamar.json",
    subcarpetas=("KNN", "8_KNN", "7_KNN"),
)
RUTA_TEST_KNN = _resolver_archivo_resultados(
    "test_final_externo_knn_multioutput_h10_calamar.csv",
    subcarpetas=("KNN", "8_KNN", "7_KNN"),
)
RUTA_MODELO_KNN = _resolver_archivo_resultados(
    "modelo_knn_multioutput_h10_calamar.joblib",
    subcarpetas=("KNN", "8_KNN", "7_KNN"),
)
RUTA_RESUMEN_KNN = _resolver_archivo_resultados(
    "resumen_knn_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("KNN", "8_KNN", "7_KNN"),
)
RUTA_RESULTADOS_CV_KNN = _resolver_archivo_resultados(
    "resultados_knn_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("KNN", "8_KNN", "7_KNN"),
)
RUTA_METRICAS_HORIZONTES_KNN = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_knn_multioutput_h10.csv",
    subcarpetas=("KNN", "8_KNN", "7_KNN"),
)

# Actualizar la entrada KNN de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "knn_calamar":
        _spec["nombre"] = "KNN Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_KNN
        _spec["ruta_test"] = RUTA_TEST_KNN


def _leer_csv_knn_modelos(ruta):
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def cargar_resultados_knn_calamar():
    with open(RUTA_METADATA_KNN, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_knn_modelos(RUTA_TEST_KNN)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_knn_multioutput():
    df_resumen = _leer_csv_knn_modelos(RUTA_RESUMEN_KNN)
    df_resultados_cv = _leer_csv_knn_modelos(RUTA_RESULTADOS_CV_KNN)
    df_metricas_horizontes = _leer_csv_knn_modelos(RUTA_METRICAS_HORIZONTES_KNN)
    return df_resumen, df_resultados_cv, df_metricas_horizontes


def layout_knn_calamar():
    metadata, df_test = cargar_resultados_knn_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes = cargar_tablas_knn_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_knn(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_knn(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)

    df_validacion = pd.DataFrame([
        {
            "Conjunto": "Train / validación interna",
            "Fecha inicial": metadata["fecha_inicio_trainval"],
            "Fecha final": metadata["fecha_fin_trainval"],
        },
        {
            "Conjunto": "Test externo final",
            "Fecha inicial": metadata["fecha_inicio_test_externo"],
            "Fecha final": metadata["fecha_fin_test_externo"],
        },
    ])

    best_params = metadata.get("best_params", {})
    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "Pipeline(StandardScaler + KNeighborsRegressor)")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "n_neighbors", "Valor": best_params.get("knn__n_neighbors", best_params.get("n_neighbors", "N/A"))},
        {"Parámetro": "weights", "Valor": best_params.get("knn__weights", best_params.get("weights", "N/A"))},
        {"Parámetro": "p", "Valor": best_params.get("knn__p", best_params.get("p", "N/A"))},
        {"Parámetro": "metric", "Valor": best_params.get("knn__metric", best_params.get("metric", "N/A"))},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(
                df_metricas_horizontes_tabla[col], errors="coerce"
            ).round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds_knn(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds_knn(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds_knn(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("K-Vecinos más Cercanos (KNN) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a un pipeline compuesto por StandardScaler y KNeighborsRegressor, "
                "configurado para predecir simultáneamente un horizonte de 10 días del nivel en la estación Calamar. "
                "La selección se realizó con validación cruzada temporal mediante split_train_val_groupKFold.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada y combinaciones de hiperparámetros del KNN. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=12),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo KNN multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_knn_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_knn_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo KNN multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


# =====================================================================
# ACTUALIZACIÓN XARIMA MULTIOUTPUT H10
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para XARIMA H10
RUTA_METADATA_XARIMA = _resolver_archivo_resultados(
    "metadata_modelo_xarima_h10_calamar.json",
    subcarpetas=("XARIMA", "9_XARIMA", "8_XARIMA"),
)
RUTA_TEST_XARIMA = _resolver_archivo_resultados(
    "test_final_externo_xarima_h10_calamar.csv",
    subcarpetas=("XARIMA", "9_XARIMA", "8_XARIMA"),
)
RUTA_MODELO_XARIMA = _resolver_archivo_resultados(
    "modelo_xarima_h10_calamar.joblib",
    subcarpetas=("XARIMA", "9_XARIMA", "8_XARIMA"),
)
RUTA_RESUMEN_XARIMA = _resolver_archivo_resultados(
    "resumen_xarima_h10_timeseries_cv_bds.csv",
    subcarpetas=("XARIMA", "9_XARIMA", "8_XARIMA"),
)
RUTA_RESULTADOS_CV_XARIMA = _resolver_archivo_resultados(
    "resultados_xarima_h10_timeseries_cv_bds.csv",
    subcarpetas=("XARIMA", "9_XARIMA", "8_XARIMA"),
)
RUTA_METRICAS_HORIZONTES_XARIMA = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_xarima_h10.csv",
    subcarpetas=("XARIMA", "9_XARIMA", "8_XARIMA"),
)

# Actualizar la entrada XARIMA de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "xarima_calamar":
        _spec["nombre"] = "XARIMA H10"
        _spec["ruta_metadata"] = RUTA_METADATA_XARIMA
        _spec["ruta_test"] = RUTA_TEST_XARIMA


def _leer_csv_xarima_modelos(ruta):
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def cargar_resultados_xarima_calamar():
    with open(RUTA_METADATA_XARIMA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_xarima_modelos(RUTA_TEST_XARIMA)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_xarima_h10():
    df_resumen = _leer_csv_xarima_modelos(RUTA_RESUMEN_XARIMA)
    df_resultados_cv = _leer_csv_xarima_modelos(RUTA_RESULTADOS_CV_XARIMA)
    df_metricas_horizontes = _leer_csv_xarima_modelos(RUTA_METRICAS_HORIZONTES_XARIMA)
    return df_resumen, df_resultados_cv, df_metricas_horizontes


def _preparar_df_bds_xarima_para_heatmap(df_resultados_cv):
    df_heatmap = df_resultados_cv.copy()
    df_heatmap.columns = df_heatmap.columns.astype(str).str.strip()

    if "ventana" not in df_heatmap.columns:
        if "ventana_cv" in df_heatmap.columns:
            df_heatmap["ventana"] = df_heatmap["ventana_cv"].astype(str)
        elif "numInputs_cv" in df_heatmap.columns:
            df_heatmap["ventana"] = df_heatmap["numInputs_cv"].astype(str) + " días"

    if "order" in df_heatmap.columns and "ventana" in df_heatmap.columns:
        df_heatmap["ventana"] = (
            df_heatmap["ventana"].astype(str)
            + " | orden "
            + df_heatmap["order"].astype(str)
        )

    return df_heatmap


def layout_xarima_calamar():
    metadata, df_test = cargar_resultados_xarima_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes = cargar_tablas_xarima_h10()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_xarima(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_xarima(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(
        _preparar_df_bds_xarima_para_heatmap(df_resultados_cv)
    )

    df_validacion = pd.DataFrame([
        {
            "Conjunto": "Train / validación interna",
            "Fecha inicial": metadata["fecha_inicio_trainval"],
            "Fecha final": metadata["fecha_fin_trainval"],
        },
        {
            "Conjunto": "Test externo final",
            "Fecha inicial": metadata["fecha_inicio_test_externo"],
            "Fecha final": metadata["fecha_fin_test_externo"],
        },
    ])

    best_params = metadata.get("best_params", {})
    order = best_params.get("order", metadata.get("order", "N/A"))
    seasonal_order = best_params.get("seasonal_order", metadata.get("seasonal_order", "N/A"))

    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "SARIMAX usado como XARIMA/SARIMA univariado")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana CV seleccionada", "Valor": f"{best_params.get('numInputs_cv', metadata.get('numInputs_cv', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "Orden ARIMA (p, d, q)", "Valor": str(tuple(order)) if isinstance(order, list) else str(order)},
        {"Parámetro": "Orden estacional", "Valor": str(tuple(seasonal_order)) if isinstance(seasonal_order, list) else str(seasonal_order)},
        {"Parámetro": "Tendencia", "Valor": best_params.get("trend", metadata.get("trend", "N/A"))},
        {"Parámetro": "Transformación log", "Valor": best_params.get("usar_log", metadata.get("usar_log", "N/A"))},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("numInputs_cv_list", metadata.get("ventanas_evaluadas", []))))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(
                df_metricas_horizontes_tabla[col], errors="coerce"
            ).round(4)

    columnas_bds = [
        "criterio",
        "criterio_bds_usado",
        "ventana_cv",
        "numInputs_cv",
        "order",
        "seasonal_order",
        "trend",
        "usar_log",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resumen.columns]
    df_bds_tabla = df_resumen[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "criterio": "Criterio",
        "criterio_bds_usado": "Criterio BDS usado",
        "ventana_cv": "Ventana CV",
        "numInputs_cv": "Entrada CV [días]",
        "order": "Orden ARIMA",
        "seasonal_order": "Orden estacional",
        "trend": "Tendencia",
        "usar_log": "Log",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds_xarima(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds_xarima(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds_xarima(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("XARIMA/ARIMA con SARIMAX - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una formulación XARIMA/ARIMA implementada mediante SARIMAX, "
                "configurada para predecir un horizonte de 10 días del nivel en la estación Calamar. "
                "La selección se realizó con validación cruzada temporal mediante split_train_val_groupKFold.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
            html.P(metadata.get("nota_metodologica", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de validación temporal y órdenes ARIMA/SARIMAX. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=12),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por configuración evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada combinación de ventana, orden ARIMA y configuración evaluada "
                "en el modelo XARIMA H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por configuración evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px","maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_xarima_h10",
                        "height":1000,
                        "width": 1000,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS de la configuración seleccionada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_xarima_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo XARIMA durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])



# =====================================================================
# ACTUALIZACIÓN CNN MULTIOUTPUT H10 + CURVA DE APRENDIZAJE
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para CNN multioutput H10
RUTA_METADATA_CNN = _resolver_archivo_resultados(
    "metadata_modelo_cnn_multioutput_h10_calamar.json",
    subcarpetas=("CNN", "8_CNN", "ResultadosCNN_MultiOutput_H10"),
)
RUTA_TEST_CNN = _resolver_archivo_resultados(
    "test_final_externo_cnn_multioutput_h10_calamar.csv",
    subcarpetas=("CNN", "8_CNN", "ResultadosCNN_MultiOutput_H10"),
)
RUTA_MODELO_CNN = _resolver_archivo_resultados(
    "modelo_cnn_multioutput_h10_calamar.keras",
    subcarpetas=("CNN", "8_CNN", "ResultadosCNN_MultiOutput_H10"),
)
RUTA_RESUMEN_CNN = _resolver_archivo_resultados(
    "resumen_cnn_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("CNN", "8_CNN", "ResultadosCNN_MultiOutput_H10"),
)
RUTA_RESULTADOS_CV_CNN = _resolver_archivo_resultados(
    "resultados_cnn_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("CNN", "8_CNN", "ResultadosCNN_MultiOutput_H10"),
)
RUTA_METRICAS_HORIZONTES_CNN = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_cnn_multioutput_h10.csv",
    subcarpetas=("CNN", "8_CNN", "ResultadosCNN_MultiOutput_H10"),
)
RUTA_HISTORIAL_ENTRENAMIENTO_CNN = _resolver_archivo_resultados(
    "historial_entrenamiento_cnn_multioutput_h10.csv",
    subcarpetas=("CNN", "8_CNN", "ResultadosCNN_MultiOutput_H10"),
)
RUTA_HISTORIAL_MODELO_FINAL_CNN = _resolver_archivo_resultados(
    "historial_modelo_final_cnn_multioutput_h10.csv",
    subcarpetas=("CNN", "8_CNN", "ResultadosCNN_MultiOutput_H10"),
)

# Actualizar la entrada CNN de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "cnn_calamar":
        _spec["nombre"] = "CNN Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_CNN
        _spec["ruta_test"] = RUTA_TEST_CNN


def cargar_resultados_cnn_calamar():
    with open(RUTA_METADATA_CNN, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_modelos(RUTA_TEST_CNN)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_cnn_multioutput():
    df_resumen = _leer_csv_modelos(RUTA_RESUMEN_CNN)
    df_resultados_cv = _leer_csv_modelos(RUTA_RESULTADOS_CV_CNN)
    df_metricas_horizontes = _leer_csv_modelos(RUTA_METRICAS_HORIZONTES_CNN)
    df_historial_entrenamiento = _leer_csv_modelos(RUTA_HISTORIAL_ENTRENAMIENTO_CNN)
    df_historial_final = _leer_csv_modelos(RUTA_HISTORIAL_MODELO_FINAL_CNN)
    return df_resumen, df_resultados_cv, df_metricas_horizontes, df_historial_entrenamiento, df_historial_final


def figura_curva_aprendizaje_cnn(df_historial_entrenamiento, df_historial_final=None, num_inputs=None):
    """Curva de aprendizaje de la CNN usando el historial de entrenamiento."""
    fig = go.Figure()

    if df_historial_entrenamiento is None or df_historial_entrenamiento.empty:
        fig.add_annotation(
            text="No hay datos disponibles para construir la curva de aprendizaje.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=420,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    df = df_historial_entrenamiento.copy()
    df.columns = df.columns.astype(str).str.strip()

    for col in ["epoch", "loss_train", "loss_val", "mae_train", "mae_val", "numInputs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if num_inputs is not None and "numInputs" in df.columns:
        df_filtrado = df[df["numInputs"] == float(num_inputs)].copy()
        if not df_filtrado.empty:
            df = df_filtrado

    columnas_media = [col for col in ["loss_train", "loss_val", "mae_train", "mae_val"] if col in df.columns]
    df_prom = df.groupby(["epoch"], as_index=False)[columnas_media].mean().sort_values("epoch")

    if "loss_train" in df_prom.columns:
        fig.add_trace(go.Scatter(
            x=df_prom["epoch"],
            y=df_prom["loss_train"],
            mode="lines",
            name="Train loss CV",
            line=dict(color=AZUL, width=2.5),
            hovertemplate="<b>Época:</b> %{x}<br><b>Loss train:</b> %{y:.5f}<extra></extra>",
        ))

    if "loss_val" in df_prom.columns:
        fig.add_trace(go.Scatter(
            x=df_prom["epoch"],
            y=df_prom["loss_val"],
            mode="lines",
            name="Validation loss CV",
            line=dict(color=CELESTE, width=2.5, dash="dash"),
            hovertemplate="<b>Época:</b> %{x}<br><b>Loss validación:</b> %{y:.5f}<extra></extra>",
        ))

    if df_historial_final is not None and not df_historial_final.empty:
        df_final = df_historial_final.copy()
        df_final.columns = df_final.columns.astype(str).str.strip()
        for col in ["epoch", "loss_train", "mae_train", "mse_train", "epochs_usados"]:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

        if "loss_train" in df_final.columns:
            fig.add_trace(go.Scatter(
                x=df_final["epoch"],
                y=df_final["loss_train"],
                mode="lines",
                name="Train loss modelo final",
                line=dict(color="#B23A48", width=2.2, dash="dot"),
                hovertemplate="<b>Época:</b> %{x}<br><b>Loss modelo final:</b> %{y:.5f}<extra></extra>",
            ))

    # Señalar explícitamente la época con menor loss de validación (val_loss)
    # usando el menor valor real del historial de validación, no el promedio por época.
    epoca_menor_val_loss = None
    valor_menor_val_loss = None
    fold_menor_val_loss = None

    if {"epoch", "loss_val"}.issubset(df.columns) and df["loss_val"].notna().any():
        idx_min_val = df["loss_val"].idxmin()
        epoca_menor_val_loss = int(df.loc[idx_min_val, "epoch"])
        valor_menor_val_loss = float(df.loc[idx_min_val, "loss_val"])
        if "fold" in df.columns and pd.notna(df.loc[idx_min_val, "fold"]):
            fold_menor_val_loss = int(df.loc[idx_min_val, "fold"])

        texto_menor_val = f"Menor val loss: época {epoca_menor_val_loss}"
        if fold_menor_val_loss is not None:
            texto_menor_val += f" | fold {fold_menor_val_loss}"

        fig.add_vline(
            x=epoca_menor_val_loss,
            line_dash="dash",
            line_width=2.5,
            line_color="#8E44AD",
            annotation_text=texto_menor_val,
            annotation_position="top left",
            annotation_font=dict(family=FUENTE, size=12, color="#8E44AD"),
        )

        fig.add_trace(go.Scatter(
            x=[epoca_menor_val_loss],
            y=[valor_menor_val_loss],
            mode="markers+text",
            name=f"Menor val loss ({epoca_menor_val_loss})",
            text=[f"mín val loss<br>época {epoca_menor_val_loss}"],
            textposition="bottom right",
            marker=dict(
                size=14,
                color="#8E44AD",
                symbol="diamond",
                line=dict(width=2, color=AZUL),
            ),
            hovertemplate=(
                "<b>Menor val loss</b><br>"
                "<b>Época:</b> %{x}<br>"
                "<b>val_loss:</b> %{y:.6f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=None,
        xaxis_title="Época",
        yaxis_title="Loss",
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
        margin=dict(l=70, r=40, t=80, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)

    return fig


def layout_cnn_calamar():
    metadata, df_test = cargar_resultados_cnn_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes, df_historial_entrenamiento, df_historial_final = cargar_tablas_cnn_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_cnn(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_cnn(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)
    fig_curva_aprendizaje = figura_curva_aprendizaje_cnn(
        df_historial_entrenamiento,
        df_historial_final,
        num_inputs=metadata.get("numInputs_seleccionado", metadata.get("numInputs")),
    )

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación interna", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo final", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata.get("best_params", {})
    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "Keras Conv1D MultiOutput Dense(10)")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "filters", "Valor": best_params.get("filters", "N/A")},
        {"Parámetro": "kernel_size", "Valor": best_params.get("kernel_size", "N/A")},
        {"Parámetro": "dropout", "Valor": best_params.get("dropout", "N/A")},
        {"Parámetro": "dense_units", "Valor": best_params.get("dense_units", "N/A")},
        {"Parámetro": "learning_rate", "Valor": best_params.get("learning_rate", "N/A")},
        {"Parámetro": "batch_size", "Valor": best_params.get("batch_size", "N/A")},
        {"Parámetro": "epochs_max", "Valor": best_params.get("epochs_max", "N/A")},
        {"Parámetro": "patience", "Valor": best_params.get("patience", "N/A")},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(df_metricas_horizontes_tabla[col], errors="coerce").round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Convolutional Neural Network (CNN) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una red convolucional 1D implementada en Keras, configurada para predecir simultáneamente "
                "un horizonte de 10 días del nivel en la estación Calamar. La selección se realizó con validación cruzada "
                "temporal mediante split_train_val_groupKFold y diagnóstico BDS de residuos.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
            html.P(metadata.get("nota_y_scaling", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada e hiperparámetros de la red CNN. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=14),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Curva de aprendizaje", style=estilo_titulo),
            html.P(
                "La curva de aprendizaje se construyó a partir del historial de entrenamiento. "
                "Se muestran las pérdidas promedio de entrenamiento y validación durante la validación cruzada, "
                "junto con la pérdida de entrenamiento del modelo final. La línea vertical resalta la época donde se obtuvo el menor loss de validación.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_curva_aprendizaje,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "curva_aprendizaje_cnn_multioutput_h10",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo CNN multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_cnn_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_cnn_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo CNN multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])



# ACTUALIZACIÓN RNN MULTIOUTPUT H10 + CURVA DE APRENDIZAJE
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para RNN multioutput H10
RUTA_METADATA_RNN = _resolver_archivo_resultados(
    "metadata_modelo_rnn_multioutput_h10_calamar.json",
    subcarpetas=("RNN", "9_RNN", "ResultadosRNN_MultiOutput_H10"),
)
RUTA_TEST_RNN = _resolver_archivo_resultados(
    "test_final_externo_rnn_multioutput_h10_calamar.csv",
    subcarpetas=("RNN", "9_RNN", "ResultadosRNN_MultiOutput_H10"),
)
RUTA_MODELO_RNN = _resolver_archivo_resultados(
    "modelo_rnn_multioutput_h10_calamar.keras",
    subcarpetas=("RNN", "9_RNN", "ResultadosRNN_MultiOutput_H10"),
)
RUTA_RESUMEN_RNN = _resolver_archivo_resultados(
    "resumen_rnn_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("RNN", "9_RNN", "ResultadosRNN_MultiOutput_H10"),
)
RUTA_RESULTADOS_CV_RNN = _resolver_archivo_resultados(
    "resultados_rnn_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("RNN", "9_RNN", "ResultadosRNN_MultiOutput_H10"),
)
RUTA_METRICAS_HORIZONTES_RNN = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_rnn_multioutput_h10.csv",
    subcarpetas=("RNN", "9_RNN", "ResultadosRNN_MultiOutput_H10"),
)
RUTA_HISTORIAL_ENTRENAMIENTO_RNN = _resolver_archivo_resultados(
    "historial_entrenamiento_rnn_multioutput_h10.csv",
    subcarpetas=("RNN", "9_RNN", "ResultadosRNN_MultiOutput_H10"),
)
RUTA_HISTORIAL_MODELO_FINAL_RNN = _resolver_archivo_resultados(
    "historial_modelo_final_rnn_multioutput_h10.csv",
    subcarpetas=("RNN", "9_RNN", "ResultadosRNN_MultiOutput_H10"),
)

# Actualizar la entrada RNN de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "rnn_calamar":
        _spec["nombre"] = "RNN Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_RNN
        _spec["ruta_test"] = RUTA_TEST_RNN


def cargar_resultados_rnn_calamar():
    with open(RUTA_METADATA_RNN, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_modelos(RUTA_TEST_RNN)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_rnn_multioutput():
    df_resumen = _leer_csv_modelos(RUTA_RESUMEN_RNN)
    df_resultados_cv = _leer_csv_modelos(RUTA_RESULTADOS_CV_RNN)
    df_metricas_horizontes = _leer_csv_modelos(RUTA_METRICAS_HORIZONTES_RNN)
    df_historial_entrenamiento = _leer_csv_modelos(RUTA_HISTORIAL_ENTRENAMIENTO_RNN)
    df_historial_final = _leer_csv_modelos(RUTA_HISTORIAL_MODELO_FINAL_RNN)
    return df_resumen, df_resultados_cv, df_metricas_horizontes, df_historial_entrenamiento, df_historial_final


def figura_curva_aprendizaje_rnn(df_historial_entrenamiento, df_historial_final=None, num_inputs=None):
    """Curva de aprendizaje de la RNN usando el historial de entrenamiento."""
    fig = go.Figure()

    if df_historial_entrenamiento is None or df_historial_entrenamiento.empty:
        fig.add_annotation(
            text="No hay datos disponibles para construir la curva de aprendizaje.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=420,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    df = df_historial_entrenamiento.copy()
    df.columns = df.columns.astype(str).str.strip()

    for col in ["epoch", "loss_train", "loss_val", "mae_train", "mae_val", "numInputs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if num_inputs is not None and "numInputs" in df.columns:
        df_filtrado = df[df["numInputs"] == float(num_inputs)].copy()
        if not df_filtrado.empty:
            df = df_filtrado

    columnas_media = [col for col in ["loss_train", "loss_val", "mae_train", "mae_val"] if col in df.columns]
    df_prom = df.groupby(["epoch"], as_index=False)[columnas_media].mean().sort_values("epoch")

    if "loss_train" in df_prom.columns:
        fig.add_trace(go.Scatter(
            x=df_prom["epoch"],
            y=df_prom["loss_train"],
            mode="lines",
            name="Train loss CV",
            line=dict(color=AZUL, width=2.5),
            hovertemplate="<b>Época:</b> %{x}<br><b>Loss train:</b> %{y:.5f}<extra></extra>",
        ))

    if "loss_val" in df_prom.columns:
        fig.add_trace(go.Scatter(
            x=df_prom["epoch"],
            y=df_prom["loss_val"],
            mode="lines",
            name="Validation loss CV",
            line=dict(color=CELESTE, width=2.5, dash="dash"),
            hovertemplate="<b>Época:</b> %{x}<br><b>Loss validación:</b> %{y:.5f}<extra></extra>",
        ))

    if df_historial_final is not None and not df_historial_final.empty:
        df_final = df_historial_final.copy()
        df_final.columns = df_final.columns.astype(str).str.strip()
        for col in ["epoch", "loss_train", "mae_train", "mse_train", "epochs_usados"]:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

        if "loss_train" in df_final.columns:
            fig.add_trace(go.Scatter(
                x=df_final["epoch"],
                y=df_final["loss_train"],
                mode="lines",
                name="Train loss modelo final",
                line=dict(color="#B23A48", width=2.2, dash="dot"),
                hovertemplate="<b>Época:</b> %{x}<br><b>Loss modelo final:</b> %{y:.5f}<extra></extra>",
            ))

    # Señalar explícitamente la época con menor loss de validación (val_loss)
    # usando el menor valor real del historial de validación, no el promedio por época.
    epoca_menor_val_loss = None
    valor_menor_val_loss = None
    fold_menor_val_loss = None

    if {"epoch", "loss_val"}.issubset(df.columns) and df["loss_val"].notna().any():
        idx_min_val = df["loss_val"].idxmin()
        epoca_menor_val_loss = int(df.loc[idx_min_val, "epoch"])
        valor_menor_val_loss = float(df.loc[idx_min_val, "loss_val"])
        if "fold" in df.columns and pd.notna(df.loc[idx_min_val, "fold"]):
            fold_menor_val_loss = int(df.loc[idx_min_val, "fold"])

        texto_menor_val = f"Menor val loss: época {epoca_menor_val_loss}"
        if fold_menor_val_loss is not None:
            texto_menor_val += f" | fold {fold_menor_val_loss}"

        fig.add_vline(
            x=epoca_menor_val_loss,
            line_dash="dash",
            line_width=2.5,
            line_color="#8E44AD",
            annotation_text=texto_menor_val,
            annotation_position="top left",
            annotation_font=dict(family=FUENTE, size=12, color="#8E44AD"),
        )

        fig.add_trace(go.Scatter(
            x=[epoca_menor_val_loss],
            y=[valor_menor_val_loss],
            mode="markers+text",
            name=f"Menor val loss ({epoca_menor_val_loss})",
            text=[f"mín val loss<br>época {epoca_menor_val_loss}"],
            textposition="bottom right",
            marker=dict(
                size=14,
                color="#8E44AD",
                symbol="diamond",
                line=dict(width=2, color=AZUL),
            ),
            hovertemplate=(
                "<b>Menor val loss</b><br>"
                "<b>Época:</b> %{x}<br>"
                "<b>val_loss:</b> %{y:.6f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=None,
        xaxis_title="Época",
        yaxis_title="Loss",
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
        margin=dict(l=70, r=40, t=80, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)

    return fig


def layout_rnn_calamar():
    metadata, df_test = cargar_resultados_rnn_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes, df_historial_entrenamiento, df_historial_final = cargar_tablas_rnn_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_rnn(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_rnn(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)
    fig_curva_aprendizaje = figura_curva_aprendizaje_rnn(
        df_historial_entrenamiento,
        df_historial_final,
        num_inputs=metadata.get("numInputs_seleccionado", metadata.get("numInputs")),
    )

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación interna", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo final", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata.get("best_params", {})
    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "Keras SimpleRNN MultiOutput Dense(10)")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "units", "Valor": best_params.get("units", "N/A")},
        {"Parámetro": "dropout", "Valor": best_params.get("dropout", "N/A")},
        {"Parámetro": "learning_rate", "Valor": best_params.get("learning_rate", "N/A")},
        {"Parámetro": "batch_size", "Valor": best_params.get("batch_size", "N/A")},
        {"Parámetro": "epochs_final", "Valor": best_params.get("epochs_final", metadata.get("epochs_final", "N/A"))},
        {"Parámetro": "epochs_max", "Valor": best_params.get("epochs_max", "N/A")},
        {"Parámetro": "patience", "Valor": best_params.get("patience", "N/A")},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(df_metricas_horizontes_tabla[col], errors="coerce").round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Recurrent Neural Network (RNN) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una red recurrente simple implementada en Keras, configurada para predecir simultáneamente "
                "un horizonte de 10 días del nivel en la estación Calamar. La selección se realizó con validación cruzada "
                "temporal mediante split_train_val_groupKFold y diagnóstico BDS de residuos.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
            html.P(metadata.get("nota_y_scaling", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada e hiperparámetros de la red RNN. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=14),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Curva de aprendizaje", style=estilo_titulo),
            html.P(
                "La curva de aprendizaje se construyó a partir del historial de entrenamiento. "
                "Se muestran las pérdidas promedio de entrenamiento y validación durante la validación cruzada, "
                "junto con la pérdida de entrenamiento del modelo final. La línea vertical resalta la época donde se obtuvo el menor loss de validación.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_curva_aprendizaje,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "curva_aprendizaje_rnn_multioutput_h10",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo RNN multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            html.P(
                "El mapa de calor resume los p-valores BDS por ventana evaluada. "
                "Los valores por encima de 0.05 indican que no se rechaza la hipótesis nula de residuos i.i.d.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_rnn_multioutput_h10",
                        "height": 380,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_rnn_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo RNN multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


# =====================================================================
# ACTUALIZACIÓN MLP MULTIOUTPUT H10 + CURVA DE APRENDIZAJE
# Bloque agregado al final para no alterar las secciones anteriores.
# =====================================================================

# Nuevas rutas para MLP multioutput H10
RUTA_METADATA_MLP = _resolver_archivo_resultados(
    "metadata_modelo_mlp_multioutput_h10_calamar.json",
    subcarpetas=("MLP", "10_MLP", "ResultadosMLP_MultiOutput_H10"),
)
RUTA_TEST_MLP = _resolver_archivo_resultados(
    "test_final_externo_mlp_multioutput_h10_calamar.csv",
    subcarpetas=("MLP", "10_MLP", "ResultadosMLP_MultiOutput_H10"),
)
RUTA_MODELO_MLP = _resolver_archivo_resultados(
    "modelo_mlp_multioutput_h10_calamar.keras",
    subcarpetas=("MLP", "10_MLP", "ResultadosMLP_MultiOutput_H10"),
)
RUTA_RESUMEN_MLP = _resolver_archivo_resultados(
    "resumen_mlp_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("MLP", "10_MLP", "ResultadosMLP_MultiOutput_H10"),
)
RUTA_RESULTADOS_CV_MLP = _resolver_archivo_resultados(
    "resultados_mlp_multioutput_h10_timeseries_cv_bds.csv",
    subcarpetas=("MLP", "10_MLP", "ResultadosMLP_MultiOutput_H10"),
)
RUTA_METRICAS_HORIZONTES_MLP = _resolver_archivo_resultados(
    "metricas_horizontes_1_5_10_mlp_multioutput_h10.csv",
    subcarpetas=("MLP", "10_MLP", "ResultadosMLP_MultiOutput_H10"),
)
RUTA_HISTORIAL_ENTRENAMIENTO_MLP = _resolver_archivo_resultados(
    "historial_entrenamiento_mlp_multioutput_h10.csv",
    subcarpetas=("MLP", "10_MLP", "ResultadosMLP_MultiOutput_H10"),
)
RUTA_HISTORIAL_MODELO_FINAL_MLP = _resolver_archivo_resultados(
    "historial_modelo_final_mlp_multioutput_h10.csv",
    subcarpetas=("MLP", "10_MLP", "ResultadosMLP_MultiOutput_H10"),
)

# Actualizar la entrada MLP de la comparación general, sin tocar las demás.
for _spec in MODELOS_COMPARACION:
    if _spec.get("codigo") == "mlp_calamar":
        _spec["nombre"] = "MLP Multioutput H10"
        _spec["ruta_metadata"] = RUTA_METADATA_MLP
        _spec["ruta_test"] = RUTA_TEST_MLP


def cargar_resultados_mlp_calamar():
    with open(RUTA_METADATA_MLP, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_test = _leer_csv_modelos(RUTA_TEST_MLP)

    if "Fecha" in df_test.columns:
        df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], errors="coerce")

    for col in ["Calamar_real", "Calamar_predicho", "Residuo", "horizonte"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if "Residuo" not in df_test.columns and {"Calamar_real", "Calamar_predicho"}.issubset(df_test.columns):
        df_test["Residuo"] = df_test["Calamar_real"] - df_test["Calamar_predicho"]

    return metadata, df_test


def cargar_tablas_mlp_multioutput():
    df_resumen = _leer_csv_modelos(RUTA_RESUMEN_MLP)
    df_resultados_cv = _leer_csv_modelos(RUTA_RESULTADOS_CV_MLP)
    df_metricas_horizontes = _leer_csv_modelos(RUTA_METRICAS_HORIZONTES_MLP)
    df_historial_entrenamiento = _leer_csv_modelos(RUTA_HISTORIAL_ENTRENAMIENTO_MLP)
    df_historial_final = _leer_csv_modelos(RUTA_HISTORIAL_MODELO_FINAL_MLP)
    return df_resumen, df_resultados_cv, df_metricas_horizontes, df_historial_entrenamiento, df_historial_final


def figura_curva_aprendizaje_mlp(df_historial_entrenamiento, df_historial_final=None, num_inputs=None):
    """Curva de aprendizaje del MLP usando el historial de entrenamiento."""
    fig = go.Figure()

    if df_historial_entrenamiento is None or df_historial_entrenamiento.empty:
        fig.add_annotation(
            text="No hay datos disponibles para construir la curva de aprendizaje.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=FUENTE, size=14, color=AZUL),
        )
        fig.update_layout(
            height=420,
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font=dict(family=FUENTE, size=13, color=AZUL),
        )
        return fig

    df = df_historial_entrenamiento.copy()
    df.columns = df.columns.astype(str).str.strip()

    for col in ["epoch", "loss_train", "loss_val", "mae_train", "mae_val", "numInputs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if num_inputs is not None and "numInputs" in df.columns:
        df_filtrado = df[df["numInputs"] == float(num_inputs)].copy()
        if not df_filtrado.empty:
            df = df_filtrado

    columnas_media = [col for col in ["loss_train", "loss_val", "mae_train", "mae_val"] if col in df.columns]
    df_prom = df.groupby(["epoch"], as_index=False)[columnas_media].mean().sort_values("epoch")

    if "loss_train" in df_prom.columns:
        fig.add_trace(go.Scatter(
            x=df_prom["epoch"],
            y=df_prom["loss_train"],
            mode="lines",
            name="Train loss CV",
            line=dict(color=AZUL, width=2.5),
            hovertemplate="<b>Época:</b> %{x}<br><b>Loss train:</b> %{y:.5f}<extra></extra>",
        ))

    if "loss_val" in df_prom.columns:
        fig.add_trace(go.Scatter(
            x=df_prom["epoch"],
            y=df_prom["loss_val"],
            mode="lines",
            name="Validation loss CV",
            line=dict(color=CELESTE, width=2.5, dash="dash"),
            hovertemplate="<b>Época:</b> %{x}<br><b>Loss validación:</b> %{y:.5f}<extra></extra>",
        ))

    if df_historial_final is not None and not df_historial_final.empty:
        df_final = df_historial_final.copy()
        df_final.columns = df_final.columns.astype(str).str.strip()
        for col in ["epoch", "loss_train", "mae_train", "mse_train", "epochs_usados"]:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

        if "loss_train" in df_final.columns:
            fig.add_trace(go.Scatter(
                x=df_final["epoch"],
                y=df_final["loss_train"],
                mode="lines",
                name="Train loss modelo final",
                line=dict(color="#B23A48", width=2.2, dash="dot"),
                hovertemplate="<b>Época:</b> %{x}<br><b>Loss modelo final:</b> %{y:.5f}<extra></extra>",
            ))

    # Señalar explícitamente la época con menor loss de validación (val_loss)
    # usando el menor valor real del historial de validación, no el promedio por época.
    if {"epoch", "loss_val"}.issubset(df.columns) and df["loss_val"].notna().any():
        idx_min_val = df["loss_val"].idxmin()
        epoca_menor_val_loss = int(df.loc[idx_min_val, "epoch"])
        valor_menor_val_loss = float(df.loc[idx_min_val, "loss_val"])
        fold_menor_val_loss = None
        if "fold" in df.columns and pd.notna(df.loc[idx_min_val, "fold"]):
            fold_menor_val_loss = int(df.loc[idx_min_val, "fold"])

        texto_menor_val = f"Menor val loss: época {epoca_menor_val_loss}"
        if fold_menor_val_loss is not None:
            texto_menor_val += f" | fold {fold_menor_val_loss}"

        fig.add_vline(
            x=epoca_menor_val_loss,
            line_dash="dash",
            line_width=2.5,
            line_color="#8E44AD",
            annotation_text=texto_menor_val,
            annotation_position="top left",
            annotation_font=dict(family=FUENTE, size=12, color="#8E44AD"),
        )

        fig.add_trace(go.Scatter(
            x=[epoca_menor_val_loss],
            y=[valor_menor_val_loss],
            mode="markers+text",
            name=f"Menor val loss ({epoca_menor_val_loss})",
            text=[f"mín val loss<br>época {epoca_menor_val_loss}"],
            textposition="bottom right",
            marker=dict(
                size=14,
                color="#8E44AD",
                symbol="diamond",
                line=dict(width=2, color=AZUL),
            ),
            hovertemplate=(
                "<b>Menor val loss</b><br>"
                "<b>Época:</b> %{x}<br>"
                "<b>val_loss:</b> %{y:.6f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=None,
        xaxis_title="Época",
        yaxis_title="Loss",
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
        margin=dict(l=70, r=40, t=80, b=60),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)

    return fig


def layout_mlp_calamar():
    metadata, df_test = cargar_resultados_mlp_calamar()
    df_resumen, df_resultados_cv, df_metricas_horizontes, df_historial_entrenamiento, df_historial_final = cargar_tablas_mlp_multioutput()

    mae = float(metadata["MAE_test_externo"])
    mse = float(metadata["MSE_test_externo"])
    rmse = float(np.sqrt(mse))

    df_serie_completa = leer_serie_completa_calamar()
    fig_particion = figura_particion_temporal_mlp(metadata, df_serie_completa, df_test)
    fig_serie = figura_serie_mlp(df_test)
    fig_hist = figura_histograma_residuos(df_test)
    nlags_acf = max(1, min(9, len(df_test["Residuo"].dropna()) - 1))
    fig_acf = figura_acf_residuos(df_test, nlags=nlags_acf)
    fig_bds_heatmap = figura_heatmap_bds_residuos(df_resultados_cv)
    fig_curva_aprendizaje = figura_curva_aprendizaje_mlp(
        df_historial_entrenamiento,
        df_historial_final,
        num_inputs=metadata.get("numInputs_seleccionado", metadata.get("numInputs")),
    )

    df_validacion = pd.DataFrame([
        {"Conjunto": "Train / validación interna", "Fecha inicial": metadata["fecha_inicio_trainval"], "Fecha final": metadata["fecha_fin_trainval"]},
        {"Conjunto": "Test externo final", "Fecha inicial": metadata["fecha_inicio_test_externo"], "Fecha final": metadata["fecha_fin_test_externo"]},
    ])

    best_params = metadata.get("best_params", {})
    df_hiper = pd.DataFrame([
        {"Parámetro": "Modelo", "Valor": metadata.get("modelo", "Keras MLP MultiOutput Dense(10)")},
        {"Parámetro": "Validación cruzada", "Valor": metadata.get("validacion_cruzada", "split_train_val_groupKFold")},
        {"Parámetro": "Ventana seleccionada", "Valor": f"{metadata.get('numInputs_seleccionado', metadata.get('numInputs', 'N/A'))} días"},
        {"Parámetro": "Horizonte de salida", "Valor": f"H{metadata.get('numOutputs', 'N/A')}"},
        {"Parámetro": "numJumps", "Valor": metadata.get("numJumps", "N/A")},
        {"Parámetro": "hidden_layers", "Valor": best_params.get("hidden_layers", best_params.get("hidden_layer_sizes", "N/A"))},
        {"Parámetro": "dropout", "Valor": best_params.get("dropout", "N/A")},
        {"Parámetro": "activation", "Valor": best_params.get("activation", "N/A")},
        {"Parámetro": "learning_rate", "Valor": best_params.get("learning_rate", best_params.get("learning_rate_init", "N/A"))},
        {"Parámetro": "batch_size", "Valor": best_params.get("batch_size", "N/A")},
        {"Parámetro": "epochs_max", "Valor": best_params.get("epochs_max", "N/A")},
        {"Parámetro": "patience", "Valor": best_params.get("patience", "N/A")},
        {"Parámetro": "Ventanas evaluadas", "Valor": ", ".join(map(str, metadata.get("ventanas_evaluadas", [])))},
    ])

    df_metricas = pd.DataFrame([
        {"Etapa": "Test externo H10", "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    ])

    df_metricas_horizontes_tabla = df_metricas_horizontes.copy()
    for col in ["MAE", "MSE", "RMSE", "R2", "MAPE_pct"]:
        if col in df_metricas_horizontes_tabla.columns:
            df_metricas_horizontes_tabla[col] = pd.to_numeric(df_metricas_horizontes_tabla[col], errors="coerce").round(4)

    columnas_bds = [
        "ventana",
        "numInputs",
        "MAE_val_h10_mean",
        "MSE_val_h10_mean",
        "BDS_pvalue_mean",
        "BDS_pvalue_min",
        "BDS_folds_pass",
        "BDS_all_folds_pass",
        "BDS_any_fold_pass",
    ]
    columnas_bds = [col for col in columnas_bds if col in df_resultados_cv.columns]
    df_bds_tabla = df_resultados_cv[columnas_bds].copy() if columnas_bds else pd.DataFrame()

    for col in ["MAE_val_h10_mean", "MSE_val_h10_mean", "BDS_pvalue_mean", "BDS_pvalue_min"]:
        if col in df_bds_tabla.columns:
            df_bds_tabla[col] = pd.to_numeric(df_bds_tabla[col], errors="coerce").round(6)

    df_bds_tabla = df_bds_tabla.rename(columns={
        "ventana": "Ventana",
        "numInputs": "Entrada [días]",
        "MAE_val_h10_mean": "MAE validación H10",
        "MSE_val_h10_mean": "MSE validación H10",
        "BDS_pvalue_mean": "BDS p-valor medio",
        "BDS_pvalue_min": "BDS p-valor mínimo",
        "BDS_folds_pass": "Folds que no rechazan H0",
        "BDS_all_folds_pass": "Todos los folds pasan",
        "BDS_any_fold_pass": "Algún fold pasa",
    })

    bds_resumen = df_resumen.iloc[0].to_dict() if len(df_resumen) > 0 else {}

    def _formato_bds(valor, decimales=6):
        if pd.isna(valor):
            return "N/A"
        try:
            return f"{float(valor):.{decimales}g}"
        except (TypeError, ValueError):
            return str(valor)

    bds_pmean_txt = _formato_bds(bds_resumen.get("BDS_pvalue_mean", np.nan))
    bds_pmin_txt = _formato_bds(bds_resumen.get("BDS_pvalue_min", np.nan))
    bds_folds_txt = str(bds_resumen.get("BDS_folds_pass", "N/A"))

    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Multi-Layer Perceptron (MLP) - Calamar", style=estilo_titulo),
            html.P(
                "Este modelo corresponde a una red neuronal MLP implementada en Keras, configurada para predecir simultáneamente "
                "un horizonte de 10 días del nivel en la estación Calamar. La selección se realizó con validación cruzada "
                "temporal mediante split_train_val_groupKFold y diagnóstico BDS de residuos.",
                style=estilo_parrafo,
            ),
            html.P(metadata.get("criterio_final", ""), style=estilo_parrafo),
            html.P(metadata.get("nota_y_scaling", ""), style=estilo_parrafo),
        ]),

        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("MAE test externo H10", f"{mae:.3f}", "Error absoluto medio"),
            tarjeta_metrica("MSE test externo H10", f"{mse:.3f}", "Error cuadrático medio"),
            tarjeta_metrica("RMSE test externo H10", f"{rmse:.3f}", "Raíz del error cuadrático medio"),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Validación temporal", style=estilo_titulo),
            html.P(
                "El modelo se entrenó con el bloque de train/validación interna y los últimos 10 registros se reservaron "
                "como test externo final, en coherencia con el horizonte H10.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_validacion, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Búsqueda y mejores hiperparámetros", style=estilo_titulo),
            html.P(
                "La búsqueda evaluó ventanas de entrada e hiperparámetros del MLP dentro de un esquema multioutput. "
                "La tabla resume la configuración seleccionada.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_hiper, page_size=14),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen de validación cruzada", style=estilo_titulo),
            html.P(
                "Esta tabla resume el criterio de selección, las métricas promedio de validación y el diagnóstico BDS "
                "obtenido durante la validación cruzada temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resumen, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resultados por ventana evaluada", style=estilo_titulo),
            html.P(
                "Se muestran los resultados de validación para cada ventana de entrada evaluada en el modelo MLP multioutput.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_resultados_cv, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Curva de aprendizaje", style=estilo_titulo),
            html.P(
                "La curva muestra la evolución del loss de entrenamiento y validación durante la validación cruzada, "
                "junto con el loss de entrenamiento del modelo final. Se señala la época en la que se obtuvo el menor val_loss.",
                style=estilo_parrafo,
            ),
            dcc.Graph(
                figure=fig_curva_aprendizaje,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "curva_aprendizaje_mlp_multioutput_h10",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Test BDS de los residuos", style=estilo_titulo),
            html.P(
                "El test BDS se usó como diagnóstico de independencia de los residuos durante la validación cruzada temporal. "
                "La hipótesis nula plantea que los residuos son independientes e idénticamente distribuidos. Por tanto, "
                "p-valores mayores o iguales a 0.05 indican que no se rechaza esa hipótesis; p-valores menores a 0.05 sugieren "
                "dependencia remanente o estructura no explicada por el modelo.",
                style=estilo_parrafo,
            ),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("BDS p-valor medio", bds_pmean_txt, "Promedio entre folds"),
                tarjeta_metrica("BDS p-valor mínimo", bds_pmin_txt, "Valor más exigente"),
                tarjeta_metrica("Folds que pasan BDS", bds_folds_txt, "p-valor ≥ 0.05"),
            ]),
            dcc.Graph(
                figure=fig_bds_heatmap,
                style={"width": "950px", "maxWidth": "100%", "margin": "0 auto", "display": "block"},
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "heatmap_bds_residuos_mlp_multioutput_h10",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
            html.P(
                "La tabla permite revisar el resultado BDS por ventana evaluada junto con las métricas de validación H10. "
                "Este diagnóstico no reemplaza las métricas predictivas, sino que complementa la selección del modelo al evaluar "
                "si los errores conservan dependencia temporal.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_bds_tabla, page_size=8),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas del test externo", style=estilo_titulo),
            crear_tabla_simple(df_metricas, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Métricas por horizonte", style=estilo_titulo),
            html.P(
                "Además del desempeño acumulado H10, se reportan las métricas para los horizontes 1, 5 y 10 días.",
                style=estilo_parrafo,
            ),
            crear_tabla_simple(df_metricas_horizontes_tabla, page_size=5),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Partición temporal del modelado", style=estilo_titulo),
            html.P(
                "La serie se dividió temporalmente en un bloque de entrenamiento y validación interna, seguido por un test externo final de 10 días. "
                "Este periodo no fue usado durante la selección de hiperparámetros.",
                style=estilo_parrafo_sec,
            ),
            dcc.Graph(
                figure=fig_particion,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "particion_temporal_mlp_multioutput_h10_calamar",
                        "height": 900,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Serie observada vs predicha", style=estilo_titulo),
            html.P(
                "La gráfica compara el nivel observado en Calamar con la predicción del modelo MLP multioutput durante los 10 días del test externo.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_serie, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),

        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagnóstico de residuos", style=estilo_titulo),
            html.P(
                "El diagnóstico de residuos permite revisar la distribución de los errores y su posible dependencia temporal. "
                "En este caso debe interpretarse considerando que el test externo tiene 10 registros.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])
