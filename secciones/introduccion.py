from dash import html

from estilos import (
    estilo_flex,
    estilo_parrafo_sec,
    estilo_tarjeta,
    estilo_titulo,
    tarjeta_metrica,
    tarjeta_texto,
)

from estilos import (
    AZUL,
    AZUL_MED,
    BLANCO,
    CELESTE,
    FUENTE,
    GRIS,
    TEXTO,
    estilo_boton_eda,
    estilo_flex,
    estilo_parrafo,
    estilo_parrafo_sec,
    estilo_tabla_data,
    estilo_tabla_header,
    estilo_tarjeta,
    estilo_titulo,
    aplicar_estilo_figura,
    boton_eda,
    crear_tabla,
    estilos_botones_activos,
    tarjeta_grafica,
    tarjeta_metrica,
    tarjeta_texto,
)



def tarjeta_mapa():
    return html.Div(style=estilo_tarjeta, children=[
        html.H2("Ubicación de la estación", style=estilo_titulo),
        
        html.P(
            "Estación Calamar — Río Magdalena, Bolívar, Colombia (10.2533° N, 74.9169° O)",
            style={**estilo_parrafo_sec, "fontSize": "18px", "marginBottom": "16px", "textAlign": "center"},
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
            "El río Magdalena es la arteria fluvial más importante de Colombia, desemboca en el mar Caribe "
            "y su cuenca que cubre cerca del 22.8% del territorio nacional. En su tramo final, conocido como el Bajo Magdalena, el río entra en "
            "una vasta llanura inundable denominada Depresión Momposina antes de llegar a su zona deltáica. "
            "La estación de Calamar, situada en el departamento de Bolívar, es considerada el "
            "ápice hidrológico de este sistema deltaico. Es el punto geográfico exacto donde se define la "
            "repartición de caudales entre el cauce principal del río Magdalena y el brazo artificial conocido como el Canal del Dique. ",
        ),
        
        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("Longitud cauce", "1536 km", " "),
            tarjeta_metrica("Caudal medio anual", "7100 m³/s", " "),
        ]),        
        
        tarjeta_mapa(),
       
        html.Div(style=estilo_tarjeta, children=[

            html.H2(
                "Objetivos",
                style=estilo_titulo
            ),

            html.P(
                "El dashboard se estructura en tres componentes principales, orientados a explorar, "
                "preprocesar y modelar la serie temporal de niveles en la estación de Calamar.",
                style=estilo_parrafo
            ),

            html.Div(
                style={
                    "display": "flex",
                    "gap": "16px",
                    "flexWrap": "wrap",
                    "marginTop": "22px",
                },
                children=[

                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "260px",
                            "backgroundColor": "white",
                            "borderRadius": "16px",
                            "padding": "22px",
                            "border": "1px solid #D9E2EC",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
                        },
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "12px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        "📊",
                                        style={
                                            "width": "42px",
                                            "height": "42px",
                                            "borderRadius": "12px",
                                            "backgroundColor": "#DDF3EA",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "fontSize": "22px",
                                            "color": "#1E8E6A",
                                            "flexShrink": "0",
                                        }
                                    ),
                                    html.H4(
                                        "Análisis exploratorio",
                                        style={
                                            "color": "#1E8E6A",
                                            "fontSize": "18px",
                                            "margin": "0px",
                                            "fontWeight": "600",
                                        }
                                    ),
                                ]
                            ),
                            html.P(
                                "Exploración de la serie temporal de nivel hídrico: distribuciones, "
                                "tendencias, estacionalidad, valores atípicos y comportamiento histórico "
                                "de las estaciones.",
                                style={
                                    **estilo_parrafo,
                                    "fontSize": "15px",
                                    "margin": "0px",
                                    "textAlign": "left",
                                    "lineHeight": "1.7",
                                }
                            ),
                        ],
                    ),

                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "260px",
                            "backgroundColor": "white",
                            "borderRadius": "16px",
                            "padding": "22px",
                            "border": "1px solid #D9E2EC",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
                        },
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "12px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        "🧩",
                                        style={
                                            "width": "42px",
                                            "height": "42px",
                                            "borderRadius": "12px",
                                            "backgroundColor": "#DFEAF6",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "fontSize": "22px",
                                            "color": "#2A6DB3",
                                            "flexShrink": "0",
                                        }
                                    ),
                                    html.H4(
                                        "Imputación de datos",
                                        style={
                                            "color": "#2A6DB3",
                                            "fontSize": "18px",
                                            "margin": "0px",
                                            "fontWeight": "600",
                                        }
                                    ),
                                ]
                            ),
                            html.P(
                                "Tratamiento de registros faltantes mediante promedio climatológico mensual "
                                "y regresión lineal múltiple.",
                                style={
                                    **estilo_parrafo,
                                    "fontSize": "15px",
                                    "margin": "0px",
                                    "textAlign": "left",
                                    "lineHeight": "1.7",
                                }
                            ),
                        ],
                    ),

                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "260px",
                            "backgroundColor": "white",
                            "borderRadius": "16px",
                            "padding": "22px",
                            "border": "1px solid #D9E2EC",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
                        },
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "12px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        "📈",
                                        style={
                                            "width": "42px",
                                            "height": "42px",
                                            "borderRadius": "12px",
                                            "backgroundColor": "#F7EAD6",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "fontSize": "22px",
                                            "color": "#B06A00",
                                            "flexShrink": "0",
                                        }
                                    ),
                                    html.H4(
                                        "Predicción de nivel",
                                        style={
                                            "color": "#A05C00",
                                            "fontSize": "18px",
                                            "margin": "0px",
                                            "fontWeight": "600",
                                        }
                                    ),
                                ]
                            ),
                            html.P(
                                "Evaluación de modelos predictivos para el nivel en Calamar, incluyendo "
                                "métricas de desempeño y diagnóstico de residuos.",
                                style={
                                    **estilo_parrafo,
                                    "fontSize": "15px",
                                    "margin": "0px",
                                    "textAlign": "left",
                                    "lineHeight": "1.7",
                                }
                            ),
                        ],
                    ),
                ]
            )
        ]),
                
        
        html.Div(style=estilo_flex, children=[
            tarjeta_metrica("Años de datos", "50", "1975 – 2025"),
            tarjeta_metrica("Modelos evaluados", "13", "Comparativa"),
        ]),
    ])
