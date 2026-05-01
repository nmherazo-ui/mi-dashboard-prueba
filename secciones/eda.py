import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, dash_table
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from datos import COLORES_ESTACIONES, NOMBRES_ESTACIONES
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


def layout_eda():
    return html.Div([
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Análisis exploratorio de datos", style=estilo_titulo),
            html.P(
                "Seleccione una de las opciones para visualizar los análisis desarrollados en la etapa exploratoria.",
                style=estilo_parrafo_sec,
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "14px",
                    "marginTop": "22px",
                },
                children=[
                    boton_eda("Exploración inicial de datos", "btn-exploracion-inicial"),
                    boton_eda("Estructura temporal", "btn-estructura-temporal"),
                    boton_eda("Correlación cruzada", "btn-correlacion-cruzada"),
                    boton_eda("Imputación de datos", "btn-imputacion-datos"),
                ],
            ),
        ]),
        html.Div(id="contenido-eda"),
    ])


def figura_series_temporales(df, columnas_estaciones):
    fig = make_subplots(
        rows=len(columnas_estaciones),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[f"Estación {NOMBRES_ESTACIONES[col]}" for col in columnas_estaciones],
    )

    for i, col in enumerate(columnas_estaciones, start=1):
        fig.add_trace(
            go.Scatter(
                x=df["Fecha"],
                y=df[col],
                mode="lines",
                name=NOMBRES_ESTACIONES[col],
                line=dict(width=1.0, color=COLORES_ESTACIONES.get(col, AZUL_MED)),
                hovertemplate=(
                    f"<b>Estación:</b> {NOMBRES_ESTACIONES[col]}<br>"
                    "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                    "<b>Nivel:</b> %{y:.2f} cm<br>"
                    "<extra></extra>"
                ),
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(
            title_text="Nivel [cm]",
            showgrid=True,
            gridcolor="#D9E2EF",
            gridwidth=0.8,
            zeroline=False,
            row=i,
            col=1,
        )
        fig.update_xaxes(
            title_text="Fecha",
            showgrid=True,
            gridcolor="#D9E2EF",
            gridwidth=0.8,
            tickformat="%Y",
            dtick="M120",
            zeroline=False,
            showticklabels=True,
            row=i,
            col=1,
        )

    fig.update_layout(
        height=320 * len(columnas_estaciones),
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font=dict(family="Georgia", size=13, color=AZUL),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=80, r=40, t=80, b=60),
    )

    for anot in fig["layout"]["annotations"]:
        anot["font"] = dict(family="Georgia", size=17, color=AZUL)
        anot["x"] = 0.5
        anot["xanchor"] = "center"

    return fig


def resumen_inicial(df):
    dimensiones = f"{df.shape[0]:,} filas × {df.shape[1]} columnas"
    fecha_min = df["Fecha"].min().strftime("%Y-%m-%d")
    fecha_max = df["Fecha"].max().strftime("%Y-%m-%d")

    df_tipos = pd.DataFrame({
        "Columna": df.dtypes.index.astype(str),
        "Tipo de dato": [
            "Temporal" if "datetime" in str(tipo)
            else "Numérico" if "float" in str(tipo) or "int" in str(tipo)
            else "Texto"
            for tipo in df.dtypes
        ],
    })

    df_head = df.head().copy()
    df_head["Fecha"] = df_head["Fecha"].dt.strftime("%Y-%m-%d")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    stats = df[num_cols].describe().T.round(2)
    stats["skewness"] = df[num_cols].skew().round(2)
    stats["kurtosis"] = df[num_cols].kurtosis().round(2)
    stats["missing%"] = (df[num_cols].isnull().mean() * 100).round(2)
    stats["coef variat"] = (df[num_cols].std() / df[num_cols].mean()).round(3)
    stats = stats.reset_index().rename(columns={
        "index": "Variable",
        "count": "Registros",
        "mean": "Media",
        "std": "Desv. estándar",
        "min": "Mínimo",
        "25%": "P25",
        "50%": "Mediana",
        "75%": "P75",
        "max": "Máximo",
        "skewness": "Asimetría",
        "kurtosis": "Curtosis",
        "missing%": "Faltantes [%]",
        "coef variat": "Coef. variación",
    })

    faltantes = df.isnull().sum().reset_index()
    faltantes.columns = ["Variable", "Datos faltantes"]
    faltantes["Porcentaje [%]"] = (faltantes["Datos faltantes"] / len(df) * 100).round(2)

    faltantes_sin_fecha = faltantes[faltantes["Variable"] != "Fecha"]
    max_faltante = faltantes_sin_fecha["Porcentaje [%]"].max()
    min_faltante = faltantes_sin_fecha["Porcentaje [%]"].min()

    return dimensiones, fecha_min, fecha_max, df_tipos, df_head, stats, faltantes, max_faltante, min_faltante


def estaciones_para_graficos(df):
    estaciones = [
        ("Calamar", "Calamar", "#d62728"),
        ("Achi", "Achi", "#1f77b4"),
        ("ElBanco", "El Banco", "#2ca02c"),
        ("SaladoBlanco", "Salado Blanco", "#9467bd"),
        ("PuertoBerrio", "Puerto Berrío", "#ff7f0e"),
        ("Barrancabermeja", "Barrancabermeja", "#19d3f3"),
    ]
    return [(col, nombre, color) for col, nombre, color in estaciones if col in df.columns]


def figura_histogramas_por_estacion(df, estaciones_hist):
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[nombre for _, nombre, _ in estaciones_hist],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )
    posiciones = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for (col, nombre, color), (fila, columna) in zip(estaciones_hist, posiciones):
        fig.add_trace(
            go.Histogram(
                x=df[col].dropna(),
                nbinsx=30,
                name=nombre,
                marker=dict(color=color),
                opacity=0.80,
                showlegend=False,
            ),
            row=fila,
            col=columna,
        )
        fig.update_xaxes(title_text="Nivel [cm]", showgrid=True, gridcolor="#D9E2EF", row=fila, col=columna)
        fig.update_yaxes(title_text="Frecuencia", showgrid=True, gridcolor="#D9E2EF", row=fila, col=columna)

    fig.update_layout(
        height=750,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Georgia", size=12, color=AZUL),
        margin=dict(l=60, r=40, t=80, b=50),
        bargap=0.08,
    )

    for anot in fig["layout"]["annotations"]:
        anot["font"] = dict(family="Georgia", size=15, color=AZUL)

    return fig


def figura_histograma_conjunto(df, estaciones_hist):
    fig = go.Figure()

    for col, nombre, color in estaciones_hist:
        serie = df[col].dropna().values
        if len(serie) < 2:
            continue

        fig.add_trace(
            go.Histogram(
                x=serie,
                nbinsx=35,
                name=nombre,
                histnorm="probability density",
                opacity=0.28,
                marker=dict(color=color),
                showlegend=True,
            )
        )

        kde = gaussian_kde(serie)
        x_grid = np.linspace(serie.min(), serie.max(), 400)
        y_kde = kde(x_grid)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_kde,
                mode="lines",
                name=nombre,
                line=dict(color=color, width=2),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=None,
        xaxis_title="Nivel [cm]",
        yaxis_title="Densidad",
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Georgia", size=13, color=AZUL),
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
        ),
        margin=dict(l=60, r=40, t=35, b=60),
        bargap=0.04,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF", range=[0, 1000])
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF")

    return fig


def figura_boxplots(df, estaciones_hist):
    fig = go.Figure()

    for col, nombre, color in estaciones_hist:
        fig.add_trace(
            go.Box(
                y=df[col].dropna(),
                name=nombre,
                marker=dict(color=color, symbol="x", size=4),
                line=dict(color=color, width=1.5),
                boxmean=True,
                hovertemplate=(
                    f"<b>Estación:</b> {nombre}<br>"
                    "<b>Nivel:</b> %{y:.2f} cm<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=None,
        yaxis_title="Nivel [cm]",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Georgia", size=13, color=AZUL),
        showlegend=False,
        margin=dict(l=60, r=40, t=30, b=80),
    )
    fig.update_xaxes(showgrid=False, tickangle=0)
    fig.update_yaxes(showgrid=True, gridcolor="#D9E2EF", zeroline=False)

    return fig


def seccion_exploracion_inicial(df, columnas_estaciones):
    fig_series = figura_series_temporales(df, columnas_estaciones)
    dimensiones, fecha_min, fecha_max, df_tipos, df_head, stats, faltantes, max_faltante, min_faltante = resumen_inicial(df)
    estaciones_hist = estaciones_para_graficos(df)
    fig_hist_estaciones = figura_histogramas_por_estacion(df, estaciones_hist)
    fig_hist_conjunto = figura_histograma_conjunto(df, estaciones_hist)
    fig_boxplots = figura_boxplots(df, estaciones_hist)

    return html.Div([
        tarjeta_texto(
            "Exploración inicial de datos",
            "Esta sección presenta una primera inspección del conjunto de datos. Se muestran las series "
            "temporales de nivel medio diario para las estaciones analizadas, junto con información básica "
            "sobre dimensiones, tipos de datos, rango temporal, estadísticas descriptivas y datos faltantes.",
        ),
        html.Div(style={**estilo_flex, "marginBottom": "24px"}, children=[
            tarjeta_metrica("Fecha inicial", fecha_min, "Primer registro"),
            tarjeta_metrica("Fecha final", fecha_max, "Último registro"),
            tarjeta_metrica("Estaciones", str(len(columnas_estaciones)), "Series analizadas"),
        ]),
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Series temporales por estación", style=estilo_titulo),
            dcc.Graph(
                figure=fig_series,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "series_temporales_estaciones",
                        "height": 1600,
                        "width": 1400,
                        "scale": 2,
                    },
                },
            ),
        ]),
        html.Div(style=estilo_flex, children=[
            html.Div(style={**estilo_tarjeta, "flex": "1"}, children=[
                html.H2("Tipos de datos", style=estilo_titulo),
                crear_tabla(df_tipos, page_size=10),
            ]),
            html.Div(style={**estilo_tarjeta, "flex": "2"}, children=[
                html.H2("Primeras filas del dataset", style=estilo_titulo),
                crear_tabla(df_head, page_size=5),
            ]),
        ]),
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Resumen estadístico de las estaciones", style=estilo_titulo),
            html.P(
                "La tabla resume las principales estadísticas descriptivas de las series numéricas. "
                "Se incluyen medidas de tendencia central, dispersión, asimetría, curtosis, porcentaje "
                "de datos faltantes y coeficiente de variación.",
                style=estilo_parrafo,
            ),
            crear_tabla(stats, page_size=10),
        ]),
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Datos faltantes por variable", style=estilo_titulo),
            html.P(
                "Esta tabla muestra el número y porcentaje de registros faltantes en cada columna del conjunto de datos. "
                "Se resalta en rojo suave la estación con mayor porcentaje de datos faltantes y en verde suave la estación "
                "con menor porcentaje de datos faltantes.",
                style=estilo_parrafo,
            ),
            crear_tabla(
                faltantes,
                page_size=10,
                style_data_conditional=[
                    {
                        "if": {"filter_query": f'{{Porcentaje [%]}} = {max_faltante}'},
                        "backgroundColor": "#F4D7D7",
                        "color": "#7A1F1F",
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": f'{{Porcentaje [%]}} = {min_faltante}'},
                        "backgroundColor": "#DDEEE3",
                        "color": "#1F5C3A",
                        "fontWeight": "bold",
                    },
                ],
            ),
        ]),
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Histograma de series de nivel", style=estilo_titulo),
            html.P(
                "A continuación se presentan los histogramas de las series de nivel por estación. "
                "Estos gráficos permiten observar la forma de la distribución de los niveles registrados "
                "en cada estación hidrológica.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist_estaciones, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Distribución conjunta de las series de nivel", style=estilo_titulo),
            html.P(
                "En este gráfico se comparan de forma conjunta las distribuciones de las series de nivel "
                "para todas las estaciones. La superposición permite identificar diferencias en rango, "
                "dispersión y concentración de valores.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_hist_conjunto, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
        html.Div(style=estilo_tarjeta, children=[
            html.H2("Diagramas de caja de las series de nivel", style=estilo_titulo),
            html.P(
                "Los diagramas de caja permiten comparar la mediana, la dispersión y la presencia "
                "de valores atípicos en las series de nivel de cada estación. Esta visualización "
                "facilita identificar diferencias en el rango y la variabilidad de los niveles registrados.",
                style=estilo_parrafo,
            ),
            dcc.Graph(figure=fig_boxplots, config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False}),
        ]),
    ])


def registrar_callbacks_eda(app, df, columnas_estaciones, serie_objetivo):
    @app.callback(
        Output("contenido-eda", "children"),
        Output("btn-exploracion-inicial", "style"),
        Output("btn-estructura-temporal", "style"),
        Output("btn-correlacion-cruzada", "style"),
        Output("btn-imputacion-datos", "style"),
        Input("btn-exploracion-inicial", "n_clicks"),
        Input("btn-estructura-temporal", "n_clicks"),
        Input("btn-correlacion-cruzada", "n_clicks"),
        Input("btn-imputacion-datos", "n_clicks"),
        prevent_initial_call=False,
    )
    def mostrar_contenido_eda(n_exploracion, n_estructura, n_correlacion, n_imputacion):
        ctx = dash.callback_context
        boton_id = "btn-exploracion-inicial" if not ctx.triggered else ctx.triggered[0]["prop_id"].split(".")[0]
        estilos = estilos_botones_activos(boton_id)

        if boton_id == "btn-exploracion-inicial":
            return seccion_exploracion_inicial(df, columnas_estaciones), *estilos

        if boton_id == "btn-estructura-temporal":
            df_temp = df.assign(Año=df["Fecha"].dt.year, Mes=df["Fecha"].dt.month)
            fig_box_mes = aplicar_estilo_figura(px.box(
                df_temp,
                x="Mes",
                y=serie_objetivo,
                title="Distribución mensual de niveles",
                color_discrete_sequence=[AZUL_MED],
            ))
            fig_anual = aplicar_estilo_figura(px.line(
                df_temp.groupby("Año", as_index=False)[serie_objetivo].mean(),
                x="Año",
                y=serie_objetivo,
                markers=True,
                title="Nivel promedio anual",
                color_discrete_sequence=[CELESTE],
            ))
            contenido = html.Div([
                tarjeta_texto(
                    "Estructura temporal",
                    "En esta sección se analiza la organización temporal de los datos. Se revisan patrones mensuales y cambios promedio entre años.",
                ),
                html.Div(style=estilo_flex, children=[
                    html.Div(style={**estilo_tarjeta, "flex": "1"}, children=[dcc.Graph(figure=fig_box_mes)]),
                    html.Div(style={**estilo_tarjeta, "flex": "1"}, children=[dcc.Graph(figure=fig_anual)]),
                ]),
            ])
            return contenido, *estilos

        if boton_id == "btn-correlacion-cruzada":
            df_lags = pd.DataFrame([
                {"Lag [días]": lag, "Correlación": df[serie_objetivo].shift(lag).corr(df[serie_objetivo])}
                for lag in range(31)
            ])
            fig_corr = aplicar_estilo_figura(px.line(
                df_lags,
                x="Lag [días]",
                y="Correlación",
                markers=True,
                title="Correlación cruzada para diferentes rezagos",
                color_discrete_sequence=[AZUL_MED],
            ))
            contenido = html.Div([
                tarjeta_grafica(
                    "Correlación cruzada",
                    "La correlación cruzada permite evaluar la relación entre una serie y versiones rezagadas de otra serie. En este ejemplo se muestra la correlación para distintos lags temporales.",
                    fig_corr,
                )
            ])
            return contenido, *estilos

        if boton_id == "btn-imputacion-datos":
            df_imputacion = pd.DataFrame({
                "Estación": ["Calamar", "Achi", "El Banco", "Salado Blanco", "Puerto Berrío", "Barrancabermeja"],
                "Prueba": ["Mann–Whitney U"] * 6,
                "p-valor": [0.421, 0.000003, 0.318, 0.274, 0.612, 0.0117],
                "¿Diferencia significativa?": ["No", "Sí", "No", "No", "No", "Sí"],
            })
            contenido = html.Div(style=estilo_tarjeta, children=[
                html.H2("Imputación de datos", style=estilo_titulo),
                html.P(
                    "La prueba Mann–Whitney U se utiliza para comparar la distribución de las series originales frente a las series imputadas. El objetivo es revisar si el proceso de imputación generó cambios estadísticamente significativos.",
                    style=estilo_parrafo,
                ),
                dash_table.DataTable(
                    data=df_imputacion.to_dict("records"),
                    columns=[{"name": col, "id": col} for col in df_imputacion.columns],
                    page_size=10,
                    style_table={"overflowX": "auto", "marginTop": "16px"},
                    style_cell={
                        "textAlign": "center",
                        "fontFamily": FUENTE,
                        "fontSize": "14px",
                        "padding": "10px",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    style_header=estilo_tabla_header,
                    style_data=estilo_tabla_data,
                    style_data_conditional=[
                        {"if": {"filter_query": '{¿Diferencia significativa?} = "Sí"'}, "backgroundColor": "#FDECEC", "fontWeight": "bold"},
                        {"if": {"filter_query": '{¿Diferencia significativa?} = "No"'}, "backgroundColor": "#EAF6EF"},
                    ],
                ),
                html.P(
                    "Nota: se considera diferencia estadísticamente significativa cuando el p-valor es menor que 0.05.",
                    style={**estilo_parrafo_sec, "fontSize": "13px", "marginTop": "14px"},
                ),
            ])
            return contenido, *estilos

        return html.Div(), *estilos
