import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, dash_table
from plotly.subplots import make_subplots
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pymannkendall as mk

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
                    boton_eda("Análisis básico de datos", "btn-correlacion-cruzada"),
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

def calcular_correlacion_cruzada(df, explicativa, target="Calamar", max_lag=30):
    resultados = []

    for lag in range(max_lag + 1):
        r = df[explicativa].shift(lag).corr(df[target])
        resultados.append({
            "Lag [días]": lag,
            "Correlación": r
        })

    out = pd.DataFrame(resultados)
    mejor = out.loc[out["Correlación"].idxmax()]

    return out, int(mejor["Lag [días]"]), float(mejor["Correlación"])

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
            # ====================================================
            # Autocorrelación
            # ====================================================
            series_cols = columnas_estaciones
            lags_test = [7, 14, 21, 30, 60, 90, 180, 365]
            max_lag = 360
            resultados_lb = []

            fig_acf = make_subplots(
                rows=2,
                cols=3,
                shared_xaxes=False,
                vertical_spacing=0.18,
                horizontal_spacing=0.10,
                subplot_titles=[
                    f"Función de autocorrelación (ACF) - {NOMBRES_ESTACIONES[col]}"
                    for col in series_cols
                ],
            )

            posiciones_acf = [
                (1, 1), (1, 2), (1, 3),
                (2, 1), (2, 2), (2, 3),
            ]

            for col, (fila, columna) in zip(series_cols, posiciones_acf):
                s = df[col].dropna()

                acf_vals = acf(
                    s,
                    nlags=max_lag,
                    fft=True,
                    missing="drop",
                )
                lags = np.arange(len(acf_vals))
                conf = 1.96 / np.sqrt(len(s))

                # Banda de confianza aproximada al 95 %
                fig_acf.add_trace(
                    go.Scatter(
                        x=np.concatenate([lags, lags[::-1]]),
                        y=np.concatenate([
                            np.full_like(lags, conf, dtype=float),
                            np.full_like(lags, -conf, dtype=float)[::-1],
                        ]),
                        fill="toself",
                        fillcolor="rgba(31, 119, 180, 0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=fila,
                    col=columna,
                )

                # Línea horizontal en cero
                fig_acf.add_trace(
                    go.Scatter(
                        x=[0, max_lag],
                        y=[0, 0],
                        mode="lines",
                        line=dict(color="#1f77b4", width=1.3),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=fila,
                    col=columna,
                )

                # Barras tipo ACF
                fig_acf.add_trace(
                    go.Bar(
                        x=lags,
                        y=acf_vals,
                        marker_color=COLORES_ESTACIONES.get(col, "#1f77b4"),
                        width=0.85,
                        opacity=0.95,
                        hovertemplate=(
                            f"<b>Serie:</b> {NOMBRES_ESTACIONES[col]}<br>"
                            "<b>Lag:</b> %{x} días<br>"
                            "<b>ACF:</b> %{y:.3f}<br>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=fila,
                    col=columna,
                )

                # Puntos en la punta para asemejar plot_acf
                fig_acf.add_trace(
                    go.Scatter(
                        x=lags,
                        y=acf_vals,
                        mode="markers",
                        marker=dict(
                            color=COLORES_ESTACIONES.get(col, "#1f77b4"),
                            size=4,
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=fila,
                    col=columna,
                )

                fig_acf.update_xaxes(
                    title_text="Lag [días]",
                    showgrid=False,
                    range=[-1, max_lag + 1],
                    row=fila,
                    col=columna,
                )
                fig_acf.update_yaxes(
                    title_text="Autocorrelación",
                    range=[-0.3, 1.05],
                    showgrid=False,
                    zeroline=False,
                    row=fila,
                    col=columna,
                )

                lb = acorr_ljungbox(
                    s,
                    lags=lags_test,
                    return_df=True,
                )
                for lag, row_lb in lb.iterrows():
                    resultados_lb.append({
                        "Serie": NOMBRES_ESTACIONES[col],
                        "Lag [días]": lag,
                        "LB_stat": round(row_lb["lb_stat"], 2),
                        "p-valor": round(row_lb["lb_pvalue"], 4),
                    })

            fig_acf.update_layout(
                height=850,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Georgia", size=12, color=AZUL),
                showlegend=False,
                barmode="overlay",
                margin=dict(l=70, r=40, t=90, b=60),
            )

            for anot in fig_acf["layout"]["annotations"]:
                anot["font"] = dict(family="Georgia", size=13, color=AZUL)
                anot["xanchor"] = "center"

            lb_df = pd.DataFrame(resultados_lb)

            # ====================================================
            # Boxplots mensuales por estación
            # ====================================================
            df_mes = df.copy()
            df_mes["Mes"] = df_mes["Fecha"].dt.month

            mes_labels = [
                "Ene", "Feb", "Mar", "Abr", "May", "Jun",
                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
            ]

            fig_boxplots_mensuales = make_subplots(
                rows=len(series_cols),
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.07,
                subplot_titles=[
                    f"Boxplots mensuales - {NOMBRES_ESTACIONES[col]}"
                    for col in series_cols
                ],
            )

            for i, col in enumerate(series_cols, start=1):
                for mes_num, mes_nombre in enumerate(mes_labels, start=1):
                    datos_mes = df_mes.loc[df_mes["Mes"] == mes_num, col].dropna()

                    fig_boxplots_mensuales.add_trace(
                        go.Box(
                            y=datos_mes,
                            name=mes_nombre,
                            marker=dict(
                                color=COLORES_ESTACIONES.get(col, AZUL_MED),
                                symbol="x",
                                size=4,
                            ),
                            line=dict(
                                color=COLORES_ESTACIONES.get(col, AZUL_MED),
                                width=1.3,
                            ),
                            boxmean=False,
                            showlegend=False,
                            hovertemplate=(
                                f"<b>Estación:</b> {NOMBRES_ESTACIONES[col]}<br>"
                                f"<b>Mes:</b> {mes_nombre}<br>"
                                "<b>Nivel:</b> %{y:.2f} cm<br>"
                                "<extra></extra>"
                            ),
                        ),
                        row=i,
                        col=1,
                    )
                    
                max_y = df_mes[col].dropna().max()
                min_y = df_mes[col].dropna().min()

                rango_y = max_y - min_y
                margen_inferior = 0.08 * rango_y
                margen_superior = 0.25 * rango_y

                fig_boxplots_mensuales.update_yaxes(
                    title_text="Nivel [cm]",
                    range=[
                        max(0, min_y - margen_inferior),
                        max_y + margen_superior
                    ],
                    showgrid=True,
                    gridcolor="#D9E2EF",
                    zeroline=False,
                    row=i,
                    col=1
                )
                
                fig_boxplots_mensuales.update_xaxes(
                    title_text="Mes",
                    showgrid=False,
                    row=i,
                    col=1,
                )

            fig_boxplots_mensuales.update_layout(
                height=420 * len(series_cols),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Georgia", size=13, color=AZUL),
                showlegend=False,
                margin=dict(l=80, r=40, t=90, b=70),
            )

            for anot in fig_boxplots_mensuales["layout"]["annotations"]:
                anot["font"] = dict(family="Georgia", size=16, color=AZUL)
                anot["x"] = 0.5
                anot["xanchor"] = "center"

            # ====================================================
            # Tendencias mensuales por estación
            # ====================================================
            df_m = df.copy()
            df_m["AñoMes"] = df_m["Fecha"].dt.to_period("M")

            df_monthly = (
                df_m.groupby("AñoMes")[series_cols]
                .mean()
                .reset_index()
            )
            df_monthly["Fecha"] = df_monthly["AñoMes"].dt.to_timestamp()

            resultados_mk = []

            fig_tendencias = make_subplots(
                rows=3,
                cols=2,
                shared_xaxes=False,
                vertical_spacing=0.14,
                horizontal_spacing=0.12,
                subplot_titles=[
                    f"Tendencia mensual - {NOMBRES_ESTACIONES[col]}"
                    for col in series_cols
                ],
            )

            posiciones_tendencias = [
                (1, 1), (1, 2),
                (2, 1), (2, 2),
                (3, 1), (3, 2),
            ]

            for col, (fila, columna) in zip(series_cols, posiciones_tendencias):
                s_mensual = df_monthly[["Fecha", col]].dropna().copy()

                x = np.arange(len(s_mensual))
                y = s_mensual[col].values

                coef = np.polyfit(x, y, 1)
                trend_line = np.poly1d(coef)(x)

                res = mk.seasonal_test(s_mensual[col], period=12)

                resultados_mk.append({
                    "Serie": NOMBRES_ESTACIONES[col],
                    "Tendencia": res.trend,
                    "p-valor": round(res.p, 4),
                    "Tau": round(res.Tau, 4),
                    "Pendiente": round(res.slope, 4),
                    "Intercepto": round(res.intercept, 4),
                })

                color_serie = COLORES_ESTACIONES.get(col, AZUL_MED)

                fig_tendencias.add_trace(
                    go.Scatter(
                        x=s_mensual["Fecha"],
                        y=s_mensual[col],
                        mode="lines",
                        name="Serie mensual",
                        line=dict(color=color_serie, width=1.3),
                        opacity=0.9,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Estación:</b> {NOMBRES_ESTACIONES[col]}<br>"
                            "<b>Fecha:</b> %{x|%Y-%m}<br>"
                            "<b>Nivel medio mensual:</b> %{y:.2f} cm<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=fila,
                    col=columna,
                )

                fig_tendencias.add_trace(
                    go.Scatter(
                        x=s_mensual["Fecha"],
                        y=trend_line,
                        mode="lines",
                        name="Tendencia lineal",
                        line=dict(color=AZUL, width=2, dash="dash"),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Tendencia:</b> {NOMBRES_ESTACIONES[col]}<br>"
                            "<b>Fecha:</b> %{x|%Y-%m}<br>"
                            "<b>Nivel estimado:</b> %{y:.2f} cm<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=fila,
                    col=columna,
                )

                fig_tendencias.update_yaxes(
                    title_text="Nivel medio mensual [cm]",
                    showgrid=True,
                    gridcolor="#D9E2EF",
                    zeroline=False,
                    row=fila,
                    col=columna,
                )

                fig_tendencias.update_xaxes(
                    title_text="Fecha",
                    showgrid=True,
                    gridcolor="#D9E2EF",
                    tickformat="%Y",
                    dtick="M120",
                    row=fila,
                    col=columna,
                )

                eje_id = (fila - 1) * 2 + columna
                xref = "x domain" if eje_id == 1 else f"x{eje_id} domain"
                yref = "y domain" if eje_id == 1 else f"y{eje_id} domain"

                fig_tendencias.add_annotation(
                    x=0.98,
                    y=0.96,
                    xref=xref,
                    yref=yref,
                    text=(
                        f"<span style='color:{color_serie};'>━━</span> Serie mensual&nbsp;&nbsp;"
                        f"<span style='color:{AZUL};'>┄┄</span> Tendencia lineal"
                    ),
                    showarrow=False,
                    align="right",
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.82)",
                    bordercolor="rgba(26,58,92,0.18)",
                    borderwidth=1,
                    font=dict(family="Georgia", size=11, color=AZUL),
                )

            fig_tendencias.update_layout(
                height=1050,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Georgia", size=12, color=AZUL),
                showlegend=False,
                margin=dict(l=80, r=40, t=95, b=70),
            )

            for anot in fig_tendencias["layout"]["annotations"]:
                texto_anotacion = getattr(anot, "text", "")

                if str(texto_anotacion).startswith("Tendencia mensual"):
                    anot.font = dict(family="Georgia", size=13, color=AZUL)
                    anot.xanchor = "center"

            mk_df = pd.DataFrame(resultados_mk)
            mk_df["Tendencia"] = mk_df["Tendencia"].replace({
                "increasing": "Creciente",
                "decreasing": "Decreciente",
                "no trend": "Sin tendencia",
            })

            contenido = html.Div([
                html.Div(style=estilo_tarjeta, children=[
                    html.P(
                        "Esta sección analiza la estructura temporal de las series de nivel. "
                        "Se evalúan patrones de dependencia temporal mediante la función de autocorrelación, "
                        "boxplots mensuales y análisis de tendencias.",
                        style=estilo_parrafo,
                    )
                ]),

                html.Div(style=estilo_tarjeta, children=[
                    html.H2("Autocorrelación", style=estilo_titulo),
                    html.P(
                        "La función de autocorrelación permite evaluar la dependencia de una serie "
                        "con sus propios valores pasados. En este caso, se calcula la autocorrelación "
                        "hasta 360 días de rezago para identificar memoria temporal y posibles patrones "
                        "asociados a ciclos hidrológicos.",
                        style=estilo_parrafo,
                    ),
                    dcc.Graph(
                        figure=fig_acf,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "autocorrelacion_acf",
                                "height": 1200,
                                "width": 1600,
                                "scale": 2,
                            },
                        },
                    ),
                ]),

                html.Div(style=estilo_tarjeta, children=[
                    html.H2("Prueba Ljung–Box", style=estilo_titulo),
                    html.P(
                        "La prueba Ljung–Box evalúa si existe autocorrelación significativa "
                        "en un conjunto de rezagos. Valores pequeños de p-valor indican evidencia "
                        "de dependencia temporal en la serie.",
                        style=estilo_parrafo,
                    ),
                    crear_tabla(lb_df, page_size=12),
                    html.P(
                        "Los resultados de la prueba Ljung–Box muestran p-valores cercanos a cero "
                        "para los rezagos evaluados en las estaciones analizadas. Esto indica que se "
                        "rechaza la hipótesis nula de ausencia de autocorrelación y confirma que las "
                        "series de nivel presentan dependencia temporal significativa. En consecuencia, "
                        "los valores pasados contienen información relevante para explicar el comportamiento "
                        "futuro de los niveles hidrométricos.",
                        style={**estilo_parrafo, "marginTop": "18px"},
                    ),
                ]),

                html.Div(style=estilo_tarjeta, children=[
                    html.H2("Boxplots mensuales por estación", style=estilo_titulo),
                    html.P(
                        "Los boxplots mensuales permiten evaluar la variabilidad de los niveles "
                        "a lo largo del año para cada estación. Esta visualización facilita la "
                        "identificación de patrones estacionales, diferencias entre meses y presencia "
                        "de valores atípicos en las series de nivel.",
                        style=estilo_parrafo,
                    ),
                    dcc.Graph(
                        figure=fig_boxplots_mensuales,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "boxplots_mensuales_estaciones",
                                "height": 1800,
                                "width": 1400,
                                "scale": 2,
                            },
                        },
                    ),
                ]),

                html.Div(style=estilo_tarjeta, children=[
                    html.H2("Tendencias mensuales por estación", style=estilo_titulo),
                    html.P(
                        "Para evaluar cambios de largo plazo, las series diarias fueron agregadas "
                        "a escala mensual. Posteriormente, se estimó una tendencia lineal sobre los "
                        "promedios mensuales y se aplicó la prueba estacional de Mann–Kendall, "
                        "considerando una periodicidad anual de 12 meses.",
                        style=estilo_parrafo,
                    ),
                    dcc.Graph(
                        figure=fig_tendencias,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "tendencias_mensuales_estaciones",
                                "height": 1800,
                                "width": 1400,
                                "scale": 2,
                            },
                        },
                    ),
                ]),

                html.Div(style=estilo_tarjeta, children=[
                    html.H2("Resultados de la prueba estacional Mann–Kendall", style=estilo_titulo),
                    html.P(
                        "La prueba estacional de Mann–Kendall permite evaluar si existe una tendencia "
                        "monótona significativa en la serie, considerando la estacionalidad mensual. "
                        "Un p-valor menor que 0.05 indica evidencia estadística de tendencia.",
                        style=estilo_parrafo,
                    ),
                    crear_tabla(
                        mk_df,
                        page_size=10,
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": '{Tendencia} = "Creciente"'
                                },
                                "backgroundColor": "#DDEEE3",
                                "color": "#1F5C3A",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {
                                    "filter_query": '{Tendencia} = "Decreciente"'
                                },
                                "backgroundColor": "#F4D7D7",
                                "color": "#7A1F1F",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {
                                    "filter_query": '{Tendencia} = "Sin tendencia"'
                                },
                                "backgroundColor": "#EAF1F8",
                                "color": "#1A3A5C",
                                "fontWeight": "bold"
                            }
                        ]
                    ),
                                       
                    html.P(
                        "Dado que las series presentaron autocorrelación significativa, la tendencia se evaluó "
                        "mediante la prueba Seasonal Mann-Kendall sobre series mensuales agregadas, en lugar "
                        "de aplicar una prueba simple sobre los datos diarios. Los resultados muestran una "
                        "tendencia creciente significativa en Calamar, Achí y El Banco, así como una tendencia "
                        "decreciente significativa en Salado Blanco y Puerto Berrío. En contraste, Barrancabermeja "
                        "no presentó una tendencia estadísticamente significativa, por lo que no se encontró "
                        "evidencia suficiente de un cambio monotónico sostenido en esa estación.",
                        style={**estilo_parrafo, "marginTop": "18px"},
                    ),
                ]),
            ])

            return contenido, *estilos

        if boton_id == "btn-correlacion-cruzada":

            # ====================================================
            # Matriz de correlación entre variables
            # ====================================================

            columnas_corr = [
                "Calamar",
                "Achi",
                "ElBanco",
                "SaladoBlanco",
                "PuertoBerrio",
                "Barrancabermeja",
            ]

            columnas_corr = [
                col for col in columnas_corr
                if col in df.columns
            ]

            nombres_corr = [
                NOMBRES_ESTACIONES[col]
                for col in columnas_corr
            ]

            corr = df[columnas_corr].corr().round(3)

            fig_corr_matriz = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=nombres_corr,
                    y=nombres_corr,
                    colorscale="Blues",
                    zmin=0,
                    zmax=1,
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                    textfont={
                        "size": 15,
                        "color": "black"
                    },
                    colorbar=dict(
                        title=dict(text=""),
                        x=1.05,
                        y=0.5,
                        len=0.82,
                        thickness=18,
                        tickfont=dict(
                            family="Georgia",
                            size=13,
                            color=AZUL
                        ),
                        ticks="outside",
                        ticklen=6,
                        tickwidth=1.5,
                        outlinewidth=1.5,
                        outlinecolor="black"
                    ),
                    hovertemplate=(
                        "<b>Variable X:</b> %{x}<br>"
                        "<b>Variable Y:</b> %{y}<br>"
                        "<b>Correlación:</b> %{z:.2f}<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig_corr_matriz.update_layout(
                title=None,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    family="Georgia",
                    size=14,
                    color=AZUL
                ),
                margin=dict(l=140, r=200, t=40, b=140),
                height=620,
                width=820
            )

            fig_corr_matriz.update_xaxes(
                tickangle=45,
                side="bottom",
                automargin=True,
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                ticks="outside",
                ticklen=8,
                tickwidth=2,
                tickcolor="black"
            )

            fig_corr_matriz.update_yaxes(
                autorange="reversed",
                automargin=True,
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                ticks="outside",
                ticklen=8,
                tickwidth=2,
                tickcolor="black"
            )

            fig_corr_matriz.add_annotation(
                text="Correlación",
                xref="paper",
                yref="paper",
                x=1.22,
                y=0.5,
                showarrow=False,
                textangle=90,
                font=dict(
                    family="Georgia",
                    size=15,
                    color=AZUL
                )
            )

            # ====================================================
            # Diagramas de dispersión entre variables
            # ====================================================

            columnas_dispersion = columnas_corr
            nombres_dispersion = {
                col: NOMBRES_ESTACIONES[col]
                for col in columnas_dispersion
            }

            df_dispersion = df[columnas_dispersion].dropna()

            n_vars = len(columnas_dispersion)

            fig_dispersion = make_subplots(
                rows=n_vars - 1,
                cols=n_vars - 1,
                horizontal_spacing=0.02,
                vertical_spacing=0.02
            )

            for i in range(1, n_vars):
                for j in range(i):

                    col_y = columnas_dispersion[i]
                    col_x = columnas_dispersion[j]

                    fig_dispersion.add_trace(
                        go.Scattergl(
                            x=df_dispersion[col_x],
                            y=df_dispersion[col_y],
                            mode="markers",
                            marker=dict(
                                size=3,
                                color="#5B86C5",
                                opacity=0.65,
                                line=dict(width=0)
                            ),
                            showlegend=False,
                            hovertemplate=(
                                f"<b>X:</b> {nombres_dispersion[col_x]}<br>"
                                f"<b>Y:</b> {nombres_dispersion[col_y]}<br>"
                                "<b>X:</b> %{x:.2f} cm<br>"
                                "<b>Y:</b> %{y:.2f} cm<br>"
                                "<extra></extra>"
                            )
                        ),
                        row=i,
                        col=j + 1
                    )

                    fig_dispersion.update_xaxes(
                        showgrid=True,
                        gridcolor="white",
                        zeroline=False,
                        showticklabels=(i == n_vars - 1),
                        title_text=nombres_dispersion[col_x] if i == n_vars - 1 else "",
                        row=i,
                        col=j + 1
                    )

                    fig_dispersion.update_yaxes(
                        showgrid=True,
                        gridcolor="white",
                        zeroline=False,
                        showticklabels=(j == 0),
                        title_text=nombres_dispersion[col_y] if j == 0 else "",
                        row=i,
                        col=j + 1
                    )

            fig_dispersion.update_layout(
                title=None,
                height=780,
                width=780,
                plot_bgcolor="#EAEAF2",
                paper_bgcolor="white",
                font=dict(
                    family="Georgia",
                    size=11,
                    color=AZUL
                ),
                margin=dict(l=50, r=20, t=5, b=50)
            )

            # ====================================================
            # Correlación cruzada
            # ====================================================

            target = "Calamar"
            max_lag = 30

            explicativas = [
                col for col in columnas_corr
                if col != target
            ]

            ccf_resultados = {}
            lags_optimos = {}
            corrs_optimas = {}

            for est in explicativas:

                ccf, lag, corr_max = calcular_correlacion_cruzada(
                    df,
                    explicativa=est,
                    target=target,
                    max_lag=max_lag
                )

                ccf_resultados[est] = ccf
                lags_optimos[est] = lag
                corrs_optimas[est] = corr_max

            df_lags = pd.DataFrame({
                "Estación": [
                    NOMBRES_ESTACIONES[e]
                    for e in explicativas
                ],
                "Lag óptimo [días]": [
                    lags_optimos[e]
                    for e in explicativas
                ],
                "Correlación máxima": [
                    round(corrs_optimas[e], 4)
                    for e in explicativas
                ]
            }).sort_values(
                "Correlación máxima",
                ascending=False
            ).reset_index(drop=True)

            n = len(explicativas)
            ncols = 2
            nrows = int(np.ceil(n / ncols))

            fig_ccf = make_subplots(
                rows=nrows,
                cols=ncols,
                shared_xaxes=False,
                vertical_spacing=0.18,
                horizontal_spacing=0.12,
                subplot_titles=[
                    f"{NOMBRES_ESTACIONES[est]} vs {NOMBRES_ESTACIONES[target]}"
                    for est in explicativas
                ]
            )

            posiciones_ccf = [
                (fila, col)
                for fila in range(1, nrows + 1)
                for col in range(1, ncols + 1)
            ]

            for est, (fila, columna) in zip(explicativas, posiciones_ccf):

                ccf = ccf_resultados[est]
                lag = lags_optimos[est]
                corr_max = corrs_optimas[est]
                color = COLORES_ESTACIONES.get(est, AZUL_MED)

                fig_ccf.add_trace(
                    go.Scatter(
                        x=ccf["Lag [días]"],
                        y=ccf["Correlación"],
                        mode="lines+markers",
                        line=dict(
                            color=color,
                            width=1.5
                        ),
                        marker=dict(
                            size=6,
                            color=color
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Estación:</b> {NOMBRES_ESTACIONES[est]}<br>"
                            "<b>Lag:</b> %{x} días<br>"
                            "<b>Correlación:</b> %{y:.3f}<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=fila,
                    col=columna
                )

                min_corr = ccf["Correlación"].min()
                max_corr = ccf["Correlación"].max()
                rango_corr = max_corr - min_corr if max_corr != min_corr else 0.02
                margen_y = 0.08 * rango_corr

                y_inf = min_corr - margen_y
                y_sup = max_corr + margen_y

                fig_ccf.add_trace(
                    go.Scatter(
                        x=ccf["Lag [días]"],
                        y=ccf["Correlación"],
                        mode="lines+markers",
                        line=dict(
                            color=color,
                            width=1.6
                        ),
                        marker=dict(
                            size=6,
                            color=color
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Estación:</b> {NOMBRES_ESTACIONES[est]}<br>"
                            "<b>Lag:</b> %{x} días<br>"
                            "<b>Correlación:</b> %{y:.3f}<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=fila,
                    col=columna
                )

                # Línea vertical punteada en el lag óptimo
                fig_ccf.add_trace(
                    go.Scatter(
                        x=[lag, lag],
                        y=[y_inf, corr_max],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=1.5,
                            dash="dash"
                        ),
                        showlegend=False,
                        hoverinfo="skip"
                    ),
                    row=fila,
                    col=columna
                )

                # Punto del máximo
                fig_ccf.add_trace(
                    go.Scatter(
                        x=[lag],
                        y=[corr_max],
                        mode="markers",
                        marker=dict(
                            size=9,
                            color=color,
                            symbol="circle"
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Lag óptimo:</b> {lag} días<br>"
                            f"<b>Correlación máxima:</b> {corr_max:.3f}<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=fila,
                    col=columna
                )

                fig_ccf.add_trace(
                    go.Scatter(
                        x=[lag],
                        y=[corr_max],
                        mode="markers",
                        marker=dict(
                            size=9,
                            color=color,
                            symbol="circle"
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Lag óptimo:</b> {lag} días<br>"
                            f"<b>Correlación máxima:</b> {corr_max:.3f}<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=fila,
                    col=columna
                )

                indice_subplot = (fila - 1) * ncols + columna

                xref_actual = "x domain" if indice_subplot == 1 else f"x{indice_subplot} domain"
                yref_actual = "y domain" if indice_subplot == 1 else f"y{indice_subplot} domain"

                fig_ccf.add_annotation(
                    x=0.04,
                    y=0.96,
                    xref=xref_actual,
                    yref=yref_actual,
                    text=f"lag óptimo = {lag}, corr = {corr_max:.2f}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.80)",
                    bordercolor="rgba(26,58,92,0.15)",
                    borderwidth=1,
                    font=dict(
                        family="Georgia",
                        size=11,
                        color=AZUL
                    )
                )

                fig_ccf.update_xaxes(
                    title_text="Lag [días]",
                    showgrid=True,
                    gridcolor="#D9E2EF",
                    range=[-0.5, max_lag + 1.5],
                    dtick=5,
                    row=fila,
                    col=columna
                )

                fig_ccf.update_yaxes(
                    title_text="Correlación",
                    showgrid=True,
                    gridcolor="#D9E2EF",
                    zeroline=False,
                    range=[y_inf, y_sup],
                    row=fila,
                    col=columna
                )

            fig_ccf.update_layout(
                height=430 * nrows,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    family="Georgia",
                    size=12,
                    color=AZUL
                ),
                showlegend=False,
                margin=dict(l=70, r=40, t=90, b=70)
            )

            for anot in fig_ccf["layout"]["annotations"]:
                anot.font = dict(
                    family="Georgia",
                    size=13,
                    color=AZUL
                )
                anot.xanchor = "center" if str(anot.text).startswith("Achi") or " vs " in str(anot.text) else anot.xanchor

            # Factor de Inflación de la Varianza (VIF)

            variables_vif = [
                "Achi",
                "ElBanco",
                "SaladoBlanco",
                "PuertoBerrio",
                "Barrancabermeja",
            ]

            variables_vif = [
                col for col in variables_vif
                if col in df.columns
            ]

            X_vif = df[variables_vif].dropna().copy()

            vif_df = pd.DataFrame({
                "Variable": [
                    NOMBRES_ESTACIONES[col]
                    for col in variables_vif
                ],
                "VIF": [
                    variance_inflation_factor(X_vif.values, i)
                    for i in range(X_vif.shape[1])
                ]
            })

            vif_df["VIF"] = vif_df["VIF"].round(3)

            vif_df["Nivel de colinealidad"] = np.select(
                [
                    vif_df["VIF"] < 5,
                    (vif_df["VIF"] >= 5) & (vif_df["VIF"] < 10),
                    vif_df["VIF"] >= 10
                ],
                [
                    "Baja",
                    "Moderada",
                    "Alta"
                ],
                default="Sin clasificar"
            )

            vif_df = vif_df.sort_values(
                "VIF",
                ascending=False
            ).reset_index(drop=True)


            contenido = html.Div([

                html.Div(style=estilo_tarjeta, children=[
                    html.P(
                        "Esta sección explora las relaciones entre variables mediante la matriz de correlación, "
                        "diagramas de dispersión, correlación cruzada y el factor de inflación de la varianza.",
                        style=estilo_parrafo,
                    )
                ]),

                html.Div(style=estilo_tarjeta, children=[

                    html.H2(
                        "Matriz de correlación entre variables",
                        style=estilo_titulo
                    ),

                    html.P(
                        "La matriz de correlación resume la relación lineal entre las series de nivel "
                        "registradas en las estaciones analizadas. Valores cercanos a 1 indican una "
                        "asociación positiva fuerte, mientras que valores cercanos a 0 sugieren una "
                        "relación lineal débil.",
                        style=estilo_parrafo
                    ),

                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center"
                        },
                        children=[
                            dcc.Graph(
                                figure=fig_corr_matriz,
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "matriz_correlacion",
                                        "height": 900,
                                        "width": 1100,
                                        "scale": 2
                                    }
                                }
                            )
                        ]
                    )

                ]),

                html.Div(style=estilo_tarjeta, children=[

                    html.H2(
                        "Diagramas de dispersión",
                        style=estilo_titulo
                    ),

                    html.P(
                        "Los diagramas de dispersión permiten observar la relación entre pares de estaciones. "
                        "Esta visualización complementa la matriz de correlación, ya que permite identificar "
                        "patrones lineales, dispersión de los datos y posibles agrupamientos o valores atípicos.",
                        style=estilo_parrafo
                    ),

                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "marginTop": "-10px"
                        },
                        children=[
                            dcc.Graph(
                                figure=fig_dispersion,
                                style={
                                    "width": "780px",
                                    "height": "780px",
                                    "margin": "0 auto",
                                    "display": "block"
                                },
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "diagramas_dispersion",
                                        "height": 1000,
                                        "width": 1000,
                                        "scale": 2
                                    }
                                }
                            )
                        ]
                    )

                ]),

                html.Div(style=estilo_tarjeta, children=[

                    html.H2(
                        "Correlación cruzada",
                        style=estilo_titulo
                    ),

                    html.P(
                        "Debido a que las estaciones explicativas se encuentran a varios cientos de kilómetros "
                        "de la estación Calamar, se espera que exista un retardo temporal entre las variables "
                        "explicativas y la serie objetivo. Por esta razón, se calcula la correlación cruzada "
                        "entre cada estación explicativa y Calamar, con el fin de identificar el desfase temporal "
                        "que maximiza la correlación.",
                        style=estilo_parrafo
                    ),

                    dcc.Graph(
                        figure=fig_ccf,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "correlacion_cruzada",
                                "height": 1200,
                                "width": 1400,
                                "scale": 2
                            }
                        }
                    )

                ]),

                html.Div(style=estilo_tarjeta, children=[

                    html.H2(
                        "Resumen de lags óptimos",
                        style=estilo_titulo
                    ),

                    html.P(
                        "La tabla resume el lag que maximiza la correlación entre cada estación explicativa "
                        "y la estación objetivo Calamar. En este análisis, un lag positivo indica que la serie "
                        "explicativa antecede temporalmente a Calamar.",
                        style=estilo_parrafo
                    ),

                    crear_tabla(
                        df_lags,
                        page_size=10
                    )

                ]),
                
                        html.Div(style=estilo_tarjeta, children=[

            html.H2(
                "Factor de Inflación de la Varianza (VIF)",
                style=estilo_titulo
            ),

            html.P(
                "El Factor de Inflación de la Varianza permite evaluar la presencia de "
                "multicolinealidad entre las variables explicativas. Valores altos de VIF "
                "indican que una variable puede estar fuertemente explicada por las demás, "
                "lo cual puede afectar la estabilidad e interpretación de algunos modelos.",
                style=estilo_parrafo
            ),

            crear_tabla(
                vif_df,
                page_size=10,
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": '{Nivel de colinealidad} = "Baja"'
                        },
                        "backgroundColor": "#DDEEE3",
                        "color": "#1F5C3A",
                        "fontWeight": "bold"
                    },
                    {
                        "if": {
                            "filter_query": '{Nivel de colinealidad} = "Moderada"'
                        },
                        "backgroundColor": "#FFF4D6",
                        "color": "#7A5A1F",
                        "fontWeight": "bold"
                    },
                    {
                        "if": {
                            "filter_query": '{Nivel de colinealidad} = "Alta"'
                        },
                        "backgroundColor": "#F4D7D7",
                        "color": "#7A1F1F",
                        "fontWeight": "bold"
                    }
                ]
            ),

            html.P(
                "Como criterio general, valores de VIF menores que 5 sugieren colinealidad baja; "
                "valores entre 5 y 10 indican colinealidad moderada; y valores superiores a 10 "
                "pueden indicar colinealidad alta entre variables explicativas.",
                style={
                    **estilo_parrafo,
                    "marginTop": "18px"
                }
            )

        ]),

            ])

            return contenido, *estilos
            

        if boton_id == "btn-imputacion-datos":

            ruta_imputados = "data/Niveles_imputados_completo.csv"

            df_imp = pd.read_csv(
                ruta_imputados,
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )

            df_imp.columns = df_imp.columns.str.strip()

            for col in df_imp.columns:
                if "Fecha" in col:
                    df_imp = df_imp.rename(columns={col: "Fecha"})

            df_imp["Fecha"] = pd.to_datetime(df_imp["Fecha"], errors="coerce")
            
            for col in df_imp.columns:
                if col != "Fecha":
                    df_imp[col] = (
                        df_imp[col]
                        .astype(str)
                        .str.replace(",", ".", regex=False)
                        .str.strip()
                    )
                    df_imp[col] = pd.to_numeric(df_imp[col], errors="coerce")
            
                        # ====================================================
            # Promedio climatológico mensual
            # ====================================================

            ruta_climatologia = "data/climatologia_mensual.csv"

            df_clima = pd.read_csv(
                ruta_climatologia,
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )

            df_clima.columns = df_clima.columns.str.strip()

            # Si el mes quedó como índice exportado desde pandas
            if "Unnamed: 0" in df_clima.columns:
                df_clima = df_clima.rename(columns={"Unnamed: 0": "Mes"})

            # Si la columna Mes no existe, se crea con 1-12
            if "Mes" not in df_clima.columns:
                df_clima.insert(0, "Mes", range(1, len(df_clima) + 1))

            meses_labels = {
                1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
            }

            df_clima["Mes"] = pd.to_numeric(df_clima["Mes"], errors="coerce")
            df_clima["Mes_nombre"] = df_clima["Mes"].map(meses_labels)

            for col in df_clima.columns:
                if col not in ["Mes", "Mes_nombre"]:
                    df_clima[col] = (
                        df_clima[col]
                        .astype(str)
                        .str.replace(",", ".", regex=False)
                        .replace("nan", np.nan)
                    )
                    df_clima[col] = pd.to_numeric(df_clima[col], errors="coerce")
                    
                estaciones_clima = [
                "Calamar",
                "Achi",
                "ElBanco",
                "SaladoBlanco",
                "PuertoBerrio",
                "Barrancabermeja",
            ]

            estaciones_clima = [
                est for est in estaciones_clima
                if est in df_clima.columns
            ]

            # Gráfica conjunta
            fig_clima_conjunta = go.Figure()

            for est in estaciones_clima:
                fig_clima_conjunta.add_trace(
                    go.Scatter(
                        x=df_clima["Mes_nombre"],
                        y=df_clima[est],
                        mode="lines+markers",
                        name=NOMBRES_ESTACIONES.get(est, est),
                        line=dict(
                            color=COLORES_ESTACIONES.get(est, AZUL_MED),
                            width=2
                        ),
                        marker=dict(size=7),
                        hovertemplate=(
                            f"<b>Estación:</b> {NOMBRES_ESTACIONES.get(est, est)}<br>"
                            "<b>Mes:</b> %{x}<br>"
                            "<b>Promedio climatológico:</b> %{y:.2f} cm<br>"
                            "<extra></extra>"
                        )
                    )
                )

            fig_clima_conjunta.update_layout(
                title=None,
                xaxis_title="Mes",
                yaxis_title="Nivel medio climatológico [cm]",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    family="Georgia",
                    size=13,
                    color=AZUL
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(26,58,92,0.18)",
                    borderwidth=1
                ),
                margin=dict(l=70, r=40, t=70, b=60),
                height=520
            )

            fig_clima_conjunta.update_xaxes(
                showgrid=True,
                gridcolor="#D9E2EF"
            )

            fig_clima_conjunta.update_yaxes(
                showgrid=True,
                gridcolor="#D9E2EF",
                zeroline=False
            )

            # Mosaico por estación
            fig_clima_panel = make_subplots(
                rows=2,
                cols=3,
                shared_xaxes=False,
                vertical_spacing=0.18,
                horizontal_spacing=0.10,
                subplot_titles=[
                    NOMBRES_ESTACIONES.get(est, est)
                    for est in estaciones_clima
                ]
            )

            posiciones_clima = [
                (1, 1), (1, 2), (1, 3),
                (2, 1), (2, 2), (2, 3)
            ]

            for est, (fila, columna) in zip(estaciones_clima, posiciones_clima):
                fig_clima_panel.add_trace(
                    go.Scatter(
                        x=df_clima["Mes_nombre"],
                        y=df_clima[est],
                        mode="lines+markers",
                        name=NOMBRES_ESTACIONES.get(est, est),
                        line=dict(
                            color=COLORES_ESTACIONES.get(est, AZUL_MED),
                            width=2
                        ),
                        marker=dict(size=6),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Estación:</b> {NOMBRES_ESTACIONES.get(est, est)}<br>"
                            "<b>Mes:</b> %{x}<br>"
                            "<b>Promedio climatológico:</b> %{y:.2f} cm<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=fila,
                    col=columna
                )

                fig_clima_panel.update_xaxes(
                    title_text="Mes",
                    showgrid=True,
                    gridcolor="#D9E2EF",
                    row=fila,
                    col=columna
                )

                fig_clima_panel.update_yaxes(
                    title_text="Nivel [cm]",
                    showgrid=True,
                    gridcolor="#D9E2EF",
                    zeroline=False,
                    row=fila,
                    col=columna
                )

            fig_clima_panel.update_layout(
                height=760,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    family="Georgia",
                    size=12,
                    color=AZUL
                ),
                showlegend=False,
                margin=dict(l=70, r=40, t=90, b=60)
            )

            for anot in fig_clima_panel["layout"]["annotations"]:
                anot.font = dict(
                    family="Georgia",
                    size=14,
                    color=AZUL
                )
                anot.xanchor = "center"

            estaciones_imp = [
                "Calamar",
                "Achi",
                "ElBanco",
                "SaladoBlanco",
                "PuertoBerrio",
                "Barrancabermeja",
            ]

            estaciones_imp = [
                col for col in estaciones_imp
                if col in df_imp.columns and f"{col}_original" in df_imp.columns
            ]

            resumen_imputacion = []

            for est in estaciones_imp:

                col_original = f"{est}_original"

                total_registros = len(df_imp)
                datos_originales = df_imp[col_original].notna().sum()
                datos_faltantes = df_imp[col_original].isna().sum()

                datos_imputados = (
                    df_imp[col_original].isna()
                    & df_imp[est].notna()
                ).sum()

                porcentaje_imputado = datos_imputados / total_registros * 100

                resumen_imputacion.append({
                    "Estación": NOMBRES_ESTACIONES.get(est, est),
                    "Registros totales": total_registros,
                    "Datos originales": int(datos_originales),
                    "Datos faltantes originales": int(datos_faltantes),
                    "Datos imputados": int(datos_imputados),
                    "Porcentaje imputado [%]": round(porcentaje_imputado, 2),
                })

            df_resumen_imp = pd.DataFrame(resumen_imputacion)
            
            max_imputados = df_resumen_imp["Datos imputados"].max()
            min_imputados = df_resumen_imp["Datos imputados"].min()

            estacion_default = estaciones_imp[0]
            col_original_default = f"{estacion_default}_original"

            mask_imp_default = (
                df_imp[col_original_default].isna()
                & df_imp[estacion_default].notna()
            )

            fig_imp = go.Figure()

            fig_imp.add_trace(
                go.Scatter(
                    x=df_imp["Fecha"],
                    y=df_imp[col_original_default],
                    mode="lines",
                    name="Serie original",
                    line=dict(color=AZUL_MED, width=1.2),
                    hovertemplate=(
                        "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                        "<b>Nivel original:</b> %{y:.2f} cm<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig_imp.add_trace(
                go.Scatter(
                    x=df_imp["Fecha"],
                    y=df_imp[estacion_default],
                    mode="lines",
                    name="Serie imputada",
                    line=dict(color="#D97B29", width=1.4, dash="dash"),
                    hovertemplate=(
                        "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                        "<b>Nivel imputado:</b> %{y:.2f} cm<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig_imp.add_trace(
                go.Scatter(
                    x=df_imp.loc[mask_imp_default, "Fecha"],
                    y=df_imp.loc[mask_imp_default, estacion_default],
                    mode="markers",
                    name="Valores imputados",
                    marker=dict(color="#B23A48", size=6, symbol="circle"),
                    hovertemplate=(
                        "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                        "<b>Valor imputado:</b> %{y:.2f} cm<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig_imp.update_layout(
                title=None,
                xaxis_title="Fecha",
                yaxis_title="Nivel [cm]",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Georgia", size=13, color=AZUL),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(26,58,92,0.18)",
                    borderwidth=1
                ),
                margin=dict(l=70, r=40, t=70, b=60),
                height=520
            )

            fig_imp.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
            
            y_vals = pd.concat([
                df_imp[col_original_default],
                df_imp[estacion_default]
            ]).dropna()

            y_min = y_vals.min()
            y_max = y_vals.max()
            margen_y = 0.08 * (y_max - y_min)

            fig_imp.update_yaxes(
                showgrid=True,
                gridcolor="#D9E2EF",
                zeroline=False,
                range=[
                    max(0, y_min - margen_y),
                    y_max + margen_y
                ]
            )
            
                        # ====================================================
            # Métricas de validación de imputación
            # ====================================================

            ruta_metricas = "data/metricas_imputacion_iterativa.csv"
            ruta_detalle_validacion = "data/detalle_validacion_imputacion_iterativa.csv"

            df_metricas_imp = pd.read_csv(
                ruta_metricas,
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )

            df_detalle_val = pd.read_csv(
                ruta_detalle_validacion,
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )

            df_metricas_imp.columns = df_metricas_imp.columns.str.strip()
            df_detalle_val.columns = df_detalle_val.columns.str.strip()

            if "Fecha" in df_detalle_val.columns:
                df_detalle_val["Fecha"] = pd.to_datetime(
                    df_detalle_val["Fecha"],
                    errors="coerce"
                )

            # Valores para resaltar en la tabla
            min_rmse = df_metricas_imp["RMSE"].min()
            max_rmse = df_metricas_imp["RMSE"].max()
            min_mape = df_metricas_imp["MAPE [%]"].min()
            max_mape = df_metricas_imp["MAPE [%]"].max()
            min_r2 = df_metricas_imp["R²"].min()
            max_r2 = df_metricas_imp["R²"].max()

            # ----------------------------------------------------
            # Gráfico comparativo de métricas
            # ----------------------------------------------------

            fig_metricas_imp = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=[
                    "RMSE",
                    "MAPE [%]",
                    "R²"
                ],
                horizontal_spacing=0.12
            )

            fig_metricas_imp.add_trace(
                go.Bar(
                    x=df_metricas_imp["Estación"],
                    y=df_metricas_imp["RMSE"],
                    name="RMSE",
                    marker_color="#5B86C5",
                    hovertemplate=(
                        "<b>Estación:</b> %{x}<br>"
                        "<b>RMSE:</b> %{y:.3f}<br>"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=1
            )

            fig_metricas_imp.add_trace(
                go.Bar(
                    x=df_metricas_imp["Estación"],
                    y=df_metricas_imp["MAPE [%]"],
                    name="MAPE [%]",
                    marker_color="#D97B29",
                    hovertemplate=(
                        "<b>Estación:</b> %{x}<br>"
                        "<b>MAPE:</b> %{y:.3f} %<br>"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=2
            )

            fig_metricas_imp.add_trace(
                go.Bar(
                    x=df_metricas_imp["Estación"],
                    y=df_metricas_imp["R²"],
                    name="R²",
                    marker_color="#3C8D5A",
                    hovertemplate=(
                        "<b>Estación:</b> %{x}<br>"
                        "<b>R²:</b> %{y:.3f}<br>"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=3
            )

            fig_metricas_imp.update_layout(
                height=520,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    family="Georgia",
                    size=12,
                    color=AZUL
                ),
                showlegend=False,
                margin=dict(l=60, r=40, t=80, b=120)
            )

            for i in range(1, 4):
                fig_metricas_imp.update_xaxes(
                    tickangle=45,
                    showgrid=False,
                    row=1,
                    col=i
                )

                fig_metricas_imp.update_yaxes(
                    showgrid=True,
                    gridcolor="#D9E2EF",
                    zeroline=False,
                    row=1,
                    col=i
                )

            for anot in fig_metricas_imp["layout"]["annotations"]:
                anot.font = dict(
                    family="Georgia",
                    size=14,
                    color=AZUL
                )

            # ----------------------------------------------------
            # Gráfico inicial real vs predicho
            # ----------------------------------------------------

            estacion_val_default = df_metricas_imp["Estación"].iloc[0]

            df_val_default = df_detalle_val[
                df_detalle_val["Estación"] == estacion_val_default
            ].copy()

            fila_metricas_default = df_metricas_imp[
                df_metricas_imp["Estación"] == estacion_val_default
            ].iloc[0]

            min_val = min(
                df_val_default["Valor real"].min(),
                df_val_default["Valor predicho"].min()
            )

            max_val = max(
                df_val_default["Valor real"].max(),
                df_val_default["Valor predicho"].max()
            )

            margen_val = 0.06 * (max_val - min_val)

            fig_val_imp = go.Figure()

            fig_val_imp.add_trace(
                go.Scatter(
                    x=df_val_default["Valor real"],
                    y=df_val_default["Valor predicho"],
                    mode="markers",
                    name="Real vs predicho",
                    marker=dict(
                        color=AZUL_MED,
                        size=6,
                        opacity=0.65
                    ),
                    hovertemplate=(
                        "<b>Valor real:</b> %{x:.2f} cm<br>"
                        "<b>Valor predicho:</b> %{y:.2f} cm<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig_val_imp.add_trace(
                go.Scatter(
                    x=[min_val - margen_val, max_val + margen_val],
                    y=[min_val - margen_val, max_val + margen_val],
                    mode="lines",
                    name="Línea ideal (y=x)",
                    line=dict(
                        color="#B23A48",
                        width=2,
                        dash="dash"
                    ),
                    hoverinfo="skip"
                )
            )

            fig_val_imp.update_layout(
                title=(
                    f"Validación: Real vs Predicho - {estacion_val_default} | "
                    f"R²={fila_metricas_default['R²']:.3f}  "
                    f"RMSE={fila_metricas_default['RMSE']:.3f}  "
                    f"MAPE={fila_metricas_default['MAPE [%]']:.3f}%"
                ),
                xaxis_title="Valor real [cm]",
                yaxis_title="Valor predicho [cm]",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    family="Georgia",
                    size=13,
                    color=AZUL
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(26,58,92,0.18)",
                    borderwidth=1
                ),
                margin=dict(l=70, r=40, t=90, b=70),
                height=560
            )

            fig_val_imp.update_xaxes(
                showgrid=True,
                gridcolor="#D9E2EF",
                range=[min_val - margen_val, max_val + margen_val]
            )

            fig_val_imp.update_yaxes(
                showgrid=True,
                gridcolor="#D9E2EF",
                zeroline=False,
                range=[min_val - margen_val, max_val + margen_val]
            )
            

            contenido = html.Div([

                html.Div(style=estilo_tarjeta, children=[

                    html.P(
                        "La imputación se realizó mediante un procedimiento iterativo apoyado en el "
                        "promedio climatológico mensual. Este promedio se utilizó como relleno auxiliar "
                        "para completar temporalmente las variables predictoras cuando estas tenían datos "
                        "faltantes en las fechas necesarias para imputar una estación objetivo.",
                        style=estilo_parrafo
                    ),

                    html.P(
                        "Posteriormente, para cada estación se ajustó un modelo de regresión lineal múltiple "
                        "usando las demás estaciones como variables explicativas. Los valores estimados por "
                        "el modelo reemplazaron progresivamente los valores auxiliares hasta obtener un "
                        "dataset final con las series completas.",
                        style=estilo_parrafo
                    ),
                ]),


                html.Div(style=estilo_tarjeta, children=[
                    html.H2("Resumen de datos imputados por estación", style=estilo_titulo),

                    html.P(
                        "La siguiente tabla resume la cantidad de datos originales disponibles, los datos "
                        "faltantes identificados en las series originales y los valores que fueron completados "
                        "en el dataset final imputado.",
                        style=estilo_parrafo
                    ),

                    crear_tabla(
                        df_resumen_imp,
                        page_size=10,
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": f"{{Datos imputados}} = {max_imputados}"
                                },
                                "backgroundColor": "#F4D7D7",
                                "color": "#7A1F1F",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {
                                    "filter_query": f"{{Datos imputados}} = {min_imputados}"
                                },
                                "backgroundColor": "#DDEEE3",
                                "color": "#1F5C3A",
                                "fontWeight": "bold"
                            }
                        ]
                    )
                ]),
                
                html.Div(style=estilo_tarjeta, children=[

                    html.H2(
                        "Promedio climatológico mensual",
                        style=estilo_titulo
                    ),

                    html.P(
                        "Antes de aplicar la imputación por regresión lineal múltiple, se calculó "
                        "el promedio climatológico mensual de cada estación. Este promedio se usó "
                        "como relleno auxiliar cuando las variables predictoras requeridas por el modelo "
                        "presentaban datos faltantes en las fechas de imputación.",
                        style=estilo_parrafo
                    ),

                    html.P(
                        "Es importante señalar que el promedio climatológico no corresponde al resultado "
                        "final de la imputación. Su función fue completar temporalmente los predictores "
                        "para permitir la estimación de los valores faltantes mediante regresión.",
                        style=estilo_parrafo
                    ),

                    dcc.Graph(
                        figure=fig_clima_conjunta,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "promedio_climatologico_conjunto",
                                "height": 900,
                                "width": 1400,
                                "scale": 2
                            }
                        }
                    ),

                    dcc.Graph(
                        figure=fig_clima_panel,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "promedio_climatologico_panel",
                                "height": 1200,
                                "width": 1400,
                                "scale": 2
                            }
                        }
                    )

                ]),

                html.Div(style=estilo_tarjeta, children=[
                    html.H2("Comparación entre serie original e imputada", style=estilo_titulo),

                    html.P(
                        "Seleccione una estación para comparar la serie original, que conserva los vacíos "
                        "del registro inicial, con la serie final imputada. Los puntos resaltados indican "
                        "las fechas donde se completaron datos faltantes.",
                        style=estilo_parrafo
                    ),

                    dcc.Dropdown(
                        id="selector-estacion-imputacion",
                        options=[
                            {
                                "label": NOMBRES_ESTACIONES.get(est, est),
                                "value": est
                            }
                            for est in estaciones_imp
                        ],
                        value=estacion_default,
                        clearable=False,
                        style={
                            "fontFamily": "Georgia",
                            "fontSize": "14px",
                            "maxWidth": "360px",
                            "marginBottom": "18px"
                        }
                    ),

                    dcc.Graph(
                        id="grafica-imputacion",
                        figure=fig_imp,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "serie_original_vs_imputada",
                                "height": 900,
                                "width": 1400,
                                "scale": 2
                            }
                        }
                    )
                ]),
                
                                html.Div(style=estilo_tarjeta, children=[

                    html.H2(
                        "Métricas de validación de la imputación",
                        style=estilo_titulo
                    ),

                    html.P(
                        "Las métricas se calcularon mediante una validación con faltantes simulados, "
                        "replicando el procedimiento iterativo de imputación apoyado en el promedio "
                        "climatológico mensual. Para cada estación se ocultó una fracción de datos "
                        "observados, se estimaron mediante regresión lineal múltiple y posteriormente "
                        "se compararon con los valores reales ocultados.",
                        style=estilo_parrafo
                    ),

                    crear_tabla(
                        df_metricas_imp,
                        page_size=10,
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": f"{{RMSE}} = {min_rmse}"
                                },
                                "backgroundColor": "#DDEEE3",
                                "color": "#1F5C3A",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {
                                    "filter_query": f"{{RMSE}} = {max_rmse}"
                                },
                                "backgroundColor": "#F4D7D7",
                                "color": "#7A1F1F",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {
                                    "filter_query": f"{{MAPE [%]}} = {min_mape}"
                                },
                                "backgroundColor": "#EAF6EF",
                                "color": "#1F5C3A",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {
                                    "filter_query": f"{{MAPE [%]}} = {max_mape}"
                                },
                                "backgroundColor": "#F9E1E1",
                                "color": "#7A1F1F",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {
                                    "filter_query": f"{{R²}} = {max_r2}"
                                },
                                "backgroundColor": "#DDEEE3",
                                "color": "#1F5C3A",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {
                                    "filter_query": f"{{R²}} = {min_r2}"
                                },
                                "backgroundColor": "#F4D7D7",
                                "color": "#7A1F1F",
                                "fontWeight": "bold"
                            }
                        ]
                    ),

                    html.P(
                        "En esta tabla, los menores valores de RMSE y MAPE indican mejor desempeño, "
                        "mientras que valores mayores de R² indican una mejor capacidad explicativa del "
                        "modelo durante la validación.",
                        style={
                            **estilo_parrafo,
                            "marginTop": "18px"
                        }
                    )

                ]),

                html.Div(style=estilo_tarjeta, children=[

                    html.H2(
                        "Comparación de métricas por estación",
                        style=estilo_titulo
                    ),

                    html.P(
                        "Los gráficos comparan el desempeño de la imputación entre estaciones. "
                        "RMSE y MAPE cuantifican el error de predicción, mientras que R² indica "
                        "qué proporción de la variabilidad observada es explicada por el modelo.",
                        style=estilo_parrafo
                    ),

                    dcc.Graph(
                        figure=fig_metricas_imp,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "metricas_imputacion",
                                "height": 900,
                                "width": 1400,
                                "scale": 2
                            }
                        }
                    )

                ]),

                html.Div(style=estilo_tarjeta, children=[

                    html.H2(
                        "Validación: real vs predicho",
                        style=estilo_titulo
                    ),

                    html.P(
                        "El gráfico compara los valores reales ocultados durante la validación con los "
                        "valores predichos por el modelo. La línea punteada representa la relación ideal "
                        "y=x; mientras más cerca estén los puntos de esta línea, mejor será el desempeño "
                        "de la imputación.",
                        style=estilo_parrafo
                    ),

                    dcc.Dropdown(
                        id="selector-estacion-validacion-imputacion",
                        options=[
                            {
                                "label": est,
                                "value": est
                            }
                            for est in df_metricas_imp["Estación"].tolist()
                        ],
                        value=estacion_val_default,
                        clearable=False,
                        style={
                            "fontFamily": "Georgia",
                            "fontSize": "14px",
                            "maxWidth": "360px",
                            "marginBottom": "18px"
                        }
                    ),

                    dcc.Graph(
                        id="grafica-validacion-imputacion",
                        figure=fig_val_imp,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "validacion_real_vs_predicho",
                                "height": 900,
                                "width": 1200,
                                "scale": 2
                            }
                        }
                    )

                ]),
                
            ])

            return contenido, *estilos

        
        @app.callback(
            Output("grafica-imputacion", "figure"),
            Input("selector-estacion-imputacion", "value"),
            prevent_initial_call=False
        )
        
        def actualizar_grafica_imputacion(estacion):

            ruta_imputados = "data/Niveles_imputados_completo.csv"
            df_imp = pd.read_csv(ruta_imputados, parse_dates=["Fecha"])

            col_original = f"{estacion}_original"

            mask_imp = (
                df_imp[col_original].isna()
                & df_imp[estacion].notna()
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_imp["Fecha"],
                    y=df_imp[col_original],
                    mode="lines",
                    name="Serie original",
                    line=dict(
                        color=AZUL_MED,
                        width=1.2
                    ),
                    hovertemplate=(
                        "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                        "<b>Nivel original:</b> %{y:.2f} cm<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_imp["Fecha"],
                    y=df_imp[estacion],
                    mode="lines",
                    name="Serie imputada",
                    line=dict(
                        color="#D97B29",
                        width=1.4,
                        dash="dash"
                    ),
                    hovertemplate=(
                        "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                        "<b>Nivel imputado:</b> %{y:.2f} cm<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_imp.loc[mask_imp, "Fecha"],
                    y=df_imp.loc[mask_imp, estacion],
                    mode="markers",
                    name="Valores imputados",
                    marker=dict(
                        color="#B23A48",
                        size=6,
                        symbol="circle"
                    ),
                    hovertemplate=(
                        "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                        "<b>Valor imputado:</b> %{y:.2f} cm<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig.update_layout(
                title=f"Serie original vs serie imputada - {NOMBRES_ESTACIONES.get(estacion, estacion)}",
                xaxis_title="Fecha",
                yaxis_title="Nivel [cm]",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    family="Georgia",
                    size=13,
                    color=AZUL
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(26,58,92,0.18)",
                    borderwidth=1
                ),
                margin=dict(l=70, r=40, t=80, b=60),
                height=520
            )

            fig.update_xaxes(
                showgrid=True,
                gridcolor="#D9E2EF"
            )

            fig.update_yaxes(
                showgrid=True,
                gridcolor="#D9E2EF",
                zeroline=False
            )

            return fig
        
        return html.Div(), *estilos

    @app.callback(
        Output("grafica-imputacion", "figure"),
        Input("selector-estacion-imputacion", "value"),
        prevent_initial_call=False
    )
    
    def actualizar_grafica_imputacion(estacion):

        ruta_imputados = "data/Niveles_imputados_completo.csv"

        df_imp = pd.read_csv(
            ruta_imputados,
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        df_imp.columns = df_imp.columns.str.strip()

        for col in df_imp.columns:
            if "Fecha" in col:
                df_imp = df_imp.rename(columns={col: "Fecha"})

        df_imp["Fecha"] = pd.to_datetime(df_imp["Fecha"], errors="coerce")

        # Convertir todas las columnas numéricas
        for col in df_imp.columns:
            if col != "Fecha":
                df_imp[col] = (
                    df_imp[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.strip()
                )
                df_imp[col] = pd.to_numeric(df_imp[col], errors="coerce")

        df_imp.columns = df_imp.columns.str.strip()

        for col in df_imp.columns:
            if "Fecha" in col:
                df_imp = df_imp.rename(columns={col: "Fecha"})

        df_imp["Fecha"] = pd.to_datetime(df_imp["Fecha"], errors="coerce")

        col_original = f"{estacion}_original"

        mask_imp = (
            df_imp[col_original].isna()
            & df_imp[estacion].notna()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_imp["Fecha"],
                y=df_imp[col_original],
                mode="lines",
                name="Serie original",
                line=dict(color=AZUL_MED, width=1.2),
                hovertemplate=(
                    "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                    "<b>Nivel original:</b> %{y:.2f} cm<br>"
                    "<extra></extra>"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_imp["Fecha"],
                y=df_imp[estacion],
                mode="lines",
                name="Serie imputada",
                line=dict(color="#D97B29", width=1.4, dash="dash"),
                hovertemplate=(
                    "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                    "<b>Nivel imputado:</b> %{y:.2f} cm<br>"
                    "<extra></extra>"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_imp.loc[mask_imp, "Fecha"],
                y=df_imp.loc[mask_imp, estacion],
                mode="markers",
                name="Valores imputados",
                marker=dict(color="#B23A48", size=6, symbol="circle"),
                hovertemplate=(
                    "<b>Fecha:</b> %{x|%Y-%m-%d}<br>"
                    "<b>Valor imputado:</b> %{y:.2f} cm<br>"
                    "<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title=f"Serie original vs serie imputada - {NOMBRES_ESTACIONES.get(estacion, estacion)}",
            xaxis_title="Fecha",
            yaxis_title="Nivel [cm]",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Georgia", size=13, color=AZUL),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(26,58,92,0.18)",
                borderwidth=1
            ),
            margin=dict(l=70, r=40, t=80, b=60),
            height=520
        )

        fig.update_xaxes(showgrid=True, gridcolor="#D9E2EF")
        y_vals = pd.concat([
            df_imp[col_original],
            df_imp[estacion]
        ]).dropna()

        y_min = y_vals.min()
        y_max = y_vals.max()
        margen_y = 0.08 * (y_max - y_min)

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#D9E2EF",
            zeroline=False,
            range=[
                max(0, y_min - margen_y),
                y_max + margen_y
            ]
        )

        return fig
    
    @app.callback(
        Output("grafica-validacion-imputacion", "figure"),
        Input("selector-estacion-validacion-imputacion", "value"),
        prevent_initial_call=False
    )
    def actualizar_grafica_validacion_imputacion(estacion):

        ruta_metricas = "data/metricas_imputacion_iterativa.csv"
        ruta_detalle_validacion = "data/detalle_validacion_imputacion_iterativa.csv"

        df_metricas_imp = pd.read_csv(
            ruta_metricas,
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        df_detalle_val = pd.read_csv(
            ruta_detalle_validacion,
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        df_metricas_imp.columns = df_metricas_imp.columns.str.strip()
        df_detalle_val.columns = df_detalle_val.columns.str.strip()

        df_val = df_detalle_val[
            df_detalle_val["Estación"] == estacion
        ].copy()

        fila_metricas = df_metricas_imp[
            df_metricas_imp["Estación"] == estacion
        ].iloc[0]

        min_val = min(
            df_val["Valor real"].min(),
            df_val["Valor predicho"].min()
        )

        max_val = max(
            df_val["Valor real"].max(),
            df_val["Valor predicho"].max()
        )

        margen_val = 0.06 * (max_val - min_val)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_val["Valor real"],
                y=df_val["Valor predicho"],
                mode="markers",
                name="Real vs predicho",
                marker=dict(
                    color=AZUL_MED,
                    size=6,
                    opacity=0.65
                ),
                hovertemplate=(
                    "<b>Valor real:</b> %{x:.2f} cm<br>"
                    "<b>Valor predicho:</b> %{y:.2f} cm<br>"
                    "<extra></extra>"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[min_val - margen_val, max_val + margen_val],
                y=[min_val - margen_val, max_val + margen_val],
                mode="lines",
                name="Línea ideal (y=x)",
                line=dict(
                    color="#B23A48",
                    width=2,
                    dash="dash"
                ),
                hoverinfo="skip"
            )
        )

        fig.update_layout(
            title=(
                f"Validación: Real vs Predicho - {estacion} | "
                f"R²={fila_metricas['R²']:.3f}  "
                f"RMSE={fila_metricas['RMSE']:.3f}  "
                f"MAPE={fila_metricas['MAPE [%]']:.3f}%"
            ),
            xaxis_title="Valor real [cm]",
            yaxis_title="Valor predicho [cm]",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(
                family="Georgia",
                size=13,
                color=AZUL
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(26,58,92,0.18)",
                borderwidth=1
            ),
            margin=dict(l=70, r=40, t=90, b=70),
            height=560
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="#D9E2EF",
            range=[min_val - margen_val, max_val + margen_val]
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#D9E2EF",
            zeroline=False,
            range=[min_val - margen_val, max_val + margen_val]
        )

        return fig