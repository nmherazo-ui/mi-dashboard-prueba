from dash import html, dcc, dash_table


# ── Colores y estilos ─────────────────────────────────────────────
AZUL, AZUL_MED, CELESTE = "#1A3A5C", "#2E6DA4", "#5BA4CF"
FONDO, BLANCO, GRIS = "#F4F7FB", "#FFFFFF", "#E8EDF3"
TEXTO, TEXTO_SEC = "#1A2B3C", "#5A7184"
LOGO_URL = "/assets/logo_un.png"
FUENTE = "'Georgia', serif"

estilo_tarjeta = {
    "background": BLANCO,
    "borderRadius": "12px",
    "padding": "28px",
    "marginBottom": "24px",
    "boxShadow": "0 2px 12px rgba(26,58,92,0.08)",
    "border": f"1px solid {GRIS}",
}

estilo_titulo = {
    "color": AZUL,
    "fontSize": "20px",
    "marginTop": 0,
    "textAlign": "center",
}

estilo_parrafo = {
    "color": TEXTO,
    "lineHeight": "1.8",
    "fontSize": "15px",
    "textAlign": "justify",
}

estilo_parrafo_sec = {
    "color": TEXTO_SEC,
    "fontSize": "14px",
    "lineHeight": "1.7",
    "textAlign": "justify",
}

estilo_flex = {
    "display": "flex",
    "gap": "20px",
}

estilo_tabla_celda = {
    "textAlign": "center",
    "fontFamily": FUENTE,
    "fontSize": "13px",
    "padding": "8px",
    "whiteSpace": "normal",
    "height": "auto",
    "minWidth": "90px",
}

estilo_tabla_header = {
    "backgroundColor": AZUL,
    "color": BLANCO,
    "fontWeight": "bold",
}

estilo_tabla_data = {
    "backgroundColor": BLANCO,
    "color": TEXTO,
    "border": f"1px solid {GRIS}",
}

estilo_tab = {
    "padding": "12px 28px",
    "fontFamily": FUENTE,
    "fontSize": "14px",
    "color": TEXTO_SEC,
    "border": "none",
    "borderBottom": "3px solid transparent",
    "background": "transparent",
    "cursor": "pointer",
}

estilo_tab_sel = {
    **estilo_tab,
    "color": AZUL,
    "borderBottom": f"3px solid {AZUL_MED}",
    "fontWeight": "bold",
}

estilo_boton_eda = {
    "background": BLANCO,
    "color": AZUL,
    "border": f"1px solid {AZUL_MED}",
    "borderRadius": "10px",
    "padding": "14px 18px",
    "fontFamily": FUENTE,
    "fontSize": "14px",
    "cursor": "pointer",
    "width": "100%",
    "boxShadow": "0 2px 8px rgba(26,58,92,0.08)",
}

estilo_boton_eda_activo = {
    **estilo_boton_eda,
    "background": AZUL_MED,
    "color": BLANCO,
    "fontWeight": "bold",
    "boxShadow": "0 6px 16px rgba(26,58,92,0.22)",
}


def tarjeta_texto(titulo, texto, estilo_texto=estilo_parrafo):
    return html.Div(style=estilo_tarjeta, children=[
        html.H2(titulo, style=estilo_titulo),
        html.P(texto, style=estilo_texto),
    ])


def tarjeta_grafica(titulo=None, texto=None, grafica=None):
    contenido = []
    if titulo:
        contenido.append(html.H2(titulo, style=estilo_titulo))
    if texto:
        contenido.append(html.P(texto, style=estilo_parrafo))
    if grafica is not None:
        contenido.append(dcc.Graph(figure=grafica))
    return html.Div(style=estilo_tarjeta, children=contenido)


def boton_eda(texto, id_boton):
    return html.Button(
        texto,
        id=id_boton,
        n_clicks=0,
        className="btn-eda",
        style=estilo_boton_eda,
    )


def tarjeta_metrica(titulo, valor, subtitulo):
    return html.Div(style={**estilo_tarjeta, "flex": "1", "textAlign": "center"}, children=[
        html.P(titulo, style={
            "color": TEXTO_SEC,
            "fontSize": "12px",
            "letterSpacing": "1px",
            "textTransform": "uppercase",
            "margin": "0 0 8px",
        }),
        html.H3(valor, style={"color": AZUL, "fontSize": "28px", "margin": "0 0 4px"}),
        html.P(subtitulo, style={"color": TEXTO_SEC, "fontSize": "12px", "margin": 0}),
    ])


def estilos_botones_activos(boton_id):
    return (
        estilo_boton_eda_activo if boton_id == "btn-exploracion-inicial" else estilo_boton_eda,
        estilo_boton_eda_activo if boton_id == "btn-estructura-temporal" else estilo_boton_eda,
        estilo_boton_eda_activo if boton_id == "btn-correlacion-cruzada" else estilo_boton_eda,
        estilo_boton_eda_activo if boton_id == "btn-imputacion-datos" else estilo_boton_eda,
    )


def aplicar_estilo_figura(fig):
    fig.update_layout(
        plot_bgcolor=BLANCO,
        paper_bgcolor=BLANCO,
        font_color=TEXTO,
        title_font_size=16,
    )
    return fig


def crear_tabla(dataframe, page_size=10, style_data_conditional=None):
    return dash_table.DataTable(
        data=dataframe.to_dict("records"),
        columns=[{"name": col, "id": col} for col in dataframe.columns],
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell=estilo_tabla_celda,
        style_header=estilo_tabla_header,
        style_data=estilo_tabla_data,
        style_data_conditional=style_data_conditional or [],
    )


def encabezado_principal():
    return html.Div(style={"background": AZUL, "padding": "32px 60px", "textAlign": "center"}, children=[
        html.Img(src=LOGO_URL, style={"height": "60px", "marginBottom": "20px"}),
        html.P("MACHINE LEARNING — PROYECTO FINAL", style={
            "color": CELESTE,
            "fontSize": "14px",
            "letterSpacing": "2px",
            "margin": "0 0 10px",
            "fontWeight": "bold",
        }),
        html.H1("Predicción de nivel en el Río Magdalena", style={
            "color": BLANCO,
            "fontSize": "32px",
            "margin": "0 0 8px",
            "fontWeight": "normal",
        }),
        html.P("Estación Hidrológica de Calamar · Datos 1975–2025", style={
            "color": CELESTE,
            "fontSize": "14px",
            "margin": "0 0 14px",
        }),
        html.P("Martha Juliana Camargo Lanza · Natalia Meza Herazo · Daniel Alejandro Rangel Infante", style={
            "color": BLANCO,
            "fontSize": "14px",
            "margin": 0,
            "opacity": "0.8",
        }),
    ])
