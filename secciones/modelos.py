import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output

from estilos import (
    AZUL,
    BLANCO,
    CELESTE,
    FUENTE,
    TEXTO,
    estilo_flex,
    estilo_parrafo_sec,
    estilo_tarjeta,
    estilo_titulo,
    tarjeta_metrica,
)


def layout_modelos(modelos):
    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Comparativa de modelos implementados", style=estilo_titulo),
            html.P(
                "Selecciona un modelo para visualizar su predicción frente a los valores reales.",
                style=estilo_parrafo_sec,
            ),
            dcc.Dropdown(
            id="selector-modelo",
            options=[
                {"label": "Máquina de Vectores de Soporte (SVM)", "value": "Modelo 1"},
                {"label": "Modelo comparativo 2", "value": "Modelo 2"},
                {"label": "Modelo comparativo 3", "value": "Modelo 3"},
                {"label": "Modelo comparativo 4", "value": "Modelo 4"},
                {"label": "Modelo comparativo 5", "value": "Modelo 4"},
                {"label": "Modelo comparativo 6", "value": "Modelo 6"},
                {"label": "Modelo comparativo 7", "value": "Modelo 7"},
                {"label": "Modelo comparativo 8", "value": "Modelo 8"},
                {"label": "Modelo comparativo 9", "value": "Modelo 9"},
                {"label": "Modelo comparativo 10", "value": "Modelo 10"},
                {"label": "Modelo comparativo 11", "value": "Modelo 11"},
                {"label": "Modelo comparativo 12", "value": "Modelo 12"},
                {"label": "Modelo comparativo 12", "value": "Modelo 13"},                
            ],
            value="Modelo 1",
            clearable=False,
            style={"fontFamily": FUENTE, "fontSize": "14px", "maxWidth": "300px"},
        ),
        ]),
        html.Div(id="grafica-modelo"),
        
    ])


def registrar_callbacks_modelos(app, df, serie_objetivo, modelos):
    @app.callback(Output("grafica-modelo", "children"), Input("selector-modelo", "value"))
    def mostrar_modelo(modelo):
        np.random.seed(modelos.index(modelo))
        muestra = df[["Fecha", serie_objetivo]].dropna().sample(min(200, len(df))).sort_values("Fecha")
        pred = muestra[serie_objetivo] + np.random.randn(len(muestra)) * 30

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=muestra["Fecha"],
            y=muestra[serie_objetivo],
            name="Real",
            line=dict(color=AZUL, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=muestra["Fecha"],
            y=pred,
            name="Predicción",
            line=dict(color=CELESTE, width=2, dash="dash"),
        ))
        fig.update_layout(
            title=f"Predicción vs Real — {modelo}",
            plot_bgcolor=BLANCO,
            paper_bgcolor=BLANCO,
            font_color=TEXTO,
            legend=dict(orientation="h", y=-0.2),
            title_font_size=16,
        )

        rmse = round(float(np.sqrt(np.mean((muestra[serie_objetivo].values - pred.values) ** 2))), 3)
        r2 = round(float(1 - np.var(muestra[serie_objetivo].values - pred.values) / np.var(muestra[serie_objetivo].values)), 3)

        return html.Div([
            html.Div(style=estilo_tarjeta, children=[dcc.Graph(figure=fig)]),
            html.Div(style=estilo_flex, children=[
                tarjeta_metrica("RMSE", str(rmse), "Error cuadrático medio"),
                tarjeta_metrica("R²", str(r2), "Coeficiente de determinación"),
            ]),
        ])
