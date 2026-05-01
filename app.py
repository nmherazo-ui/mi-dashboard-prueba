import dash
from dash import html, dcc, Input, Output

from datos import cargar_datos
from estilos import FONDO, FUENTE, GRIS, encabezado_principal, estilo_tab, estilo_tab_sel
from secciones.introduccion import layout_introduccion
from secciones.eda import layout_eda, registrar_callbacks_eda
from secciones.modelos import layout_modelos, registrar_callbacks_modelos


app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


df, columnas_estaciones, serie_objetivo = cargar_datos()
modelos = [f"Modelo {i}" for i in range(1, 14)]


app.layout = html.Div(
    style={"background": FONDO, "minHeight": "100vh", "fontFamily": FUENTE},
    children=[
        encabezado_principal(),
        html.Div(
            style={"maxWidth": "1100px", "margin": "0 auto", "padding": "32px 32px 60px"},
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="introduccion",
                    style={"borderBottom": f"1px solid {GRIS}", "marginBottom": "32px"},
                    children=[
                        dcc.Tab(
                            label="Introducción",
                            value="introduccion",
                            style=estilo_tab,
                            selected_style=estilo_tab_sel,
                        ),
                        dcc.Tab(
                            label="Análisis exploratorio (EDA)",
                            value="exploratorio",
                            style=estilo_tab,
                            selected_style=estilo_tab_sel,
                        ),
                        dcc.Tab(
                            label="Modelos implementados",
                            value="modelos",
                            style=estilo_tab,
                            selected_style=estilo_tab_sel,
                        ),
                    ],
                ),
                html.Div(id="contenido-tab"),
            ],
        ),
    ],
)


@app.callback(Output("contenido-tab", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "introduccion":
        return layout_introduccion()

    if tab == "exploratorio":
        return layout_eda()

    if tab == "modelos":
        return layout_modelos(modelos)

    return html.Div()


registrar_callbacks_eda(app, df, columnas_estaciones, serie_objetivo)
registrar_callbacks_modelos(app, df, serie_objetivo, modelos)


if __name__ == "__main__":
    app.run(debug=True)
