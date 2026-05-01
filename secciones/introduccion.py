from dash import html

from estilos import (
    estilo_flex,
    estilo_parrafo_sec,
    estilo_tarjeta,
    estilo_titulo,
    tarjeta_metrica,
    tarjeta_texto,
)


def tarjeta_mapa():
    return html.Div(style=estilo_tarjeta, children=[
        html.H2("Ubicación de la estación", style=estilo_titulo),
        html.P(
            "Estación Calamar — Río Magdalena, Bolívar, Colombia (10.2533° N, 74.9169° O)",
            style={**estilo_parrafo_sec, "fontSize": "13px", "marginBottom": "16px", "textAlign": "center"},
        ),
        html.Div(style={"display": "flex", "justifyContent": "center", "alignItems": "center"}, children=[
            html.Img(
                src="/assets/MapaEstaciones.png",
                style={"width": "70%", "borderRadius": "20px", "display": "block"},
            )
        ]),
    ])


def layout_introduccion():
    return html.Div([
        tarjeta_texto(
            "Contexto del estudio",
            "El río Magdalena es el principal sistema fluvial de Colombia. La estación de Calamar, "
            "ubicada en el departamento de Bolívar, registra niveles hidrológicos críticos que afectan "
            "a comunidades ribereñas, la navegación fluvial y la gestión del riesgo de inundaciones.",
        ),
        tarjeta_mapa(),
        tarjeta_texto(
            "Objetivo",
            "Desarrollar y comparar 13 modelos de predicción del nivel del río Magdalena en la estación "
            "de Calamar, evaluando su precisión y aplicabilidad para la toma de decisiones en gestión hidrológica.",
        ),
        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("Años de datos", "50", "1975 – 2025"),
            tarjeta_metrica("Modelos evaluados", "13", "Comparativa"),
            tarjeta_metrica("Estación", "Calamar", "Bolívar, Colombia"),
        ]),
    ])
