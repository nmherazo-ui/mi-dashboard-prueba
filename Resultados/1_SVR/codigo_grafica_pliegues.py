import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# Ruta del CSV de pliegues reales
# ============================================================

ruta_csv = Path(
    r"C:\Users\Victus\Documents\Academico\MCT\Semestre2\MachineLearning\PF\src_Entregable2\Modelos\ModelosCorregidos\Resultados\plotly_pliegues_reales_mejor_ventana_svr_multioutput_h10.csv"
)

archivo_html = ruta_csv.with_suffix(".html")

# ============================================================
# Cargar CSV
# ============================================================

df = pd.read_csv(ruta_csv)

print("Columnas del CSV:")
print(df.columns.tolist())

print("\nVentanas disponibles:")
print(df["numInputs"].unique())

print("\nTipos de segmentos:")
print(df["tipo_segmento"].unique())

# ============================================================
# Seleccionar ventana
# ============================================================

# Si el CSV es el de la mejor ventana, normalmente solo habrá un valor.
num_inputs = int(df["numInputs"].iloc[0])

df_plot = df[df["numInputs"] == num_inputs].copy()

# ============================================================
# Configuración visual
# ============================================================

colores = {
    "X": "blue",
    "y": "lightblue",
    "Xcv": "red",
    "ycv": "tomato",
}

nombres = {
    "X": "X (Train input)",
    "y": "y (Train output)",
    "Xcv": "Xcv (Cross-validation input)",
    "ycv": "ycv (Cross-validation output)",
}

orden = ["X", "y", "Xcv", "ycv"]

folds = sorted(df_plot["fold"].unique())

# Longitud máxima del eje x según el CSV
n_total = int(df_plot["x_fin"].max()) + 1

# ============================================================
# Crear figura Plotly
# ============================================================

fig = go.Figure()

# Línea negra superior: serie original
fig.add_trace(
    go.Scattergl(
        x=[0, n_total - 1],
        y=[0.35, 0.35],
        mode="lines",
        line=dict(color="black", width=3),
        name="Original Time-Series",
        hoverinfo="skip"
    )
)

# Áreas sombreadas por fold
for fold in folds:
    y_centro = fold + 1

    fig.add_shape(
        type="rect",
        x0=0,
        x1=n_total - 1,
        y0=y_centro - 0.42,
        y1=y_centro + 0.42,
        line=dict(color="black", width=1, dash="dash"),
        fillcolor="lavender",
        opacity=0.55,
        layer="below"
    )

# Segmentos X, y, Xcv, ycv
for tipo in orden:
    for fold in folds:
        sub = df_plot[
            (df_plot["tipo_segmento"] == tipo) &
            (df_plot["fold"] == fold)
        ].copy()

        if sub.empty:
            continue

        x_vals = []
        y_vals = []
        text_vals = []

        for _, row in sub.iterrows():
            x_vals += [row["x_inicio"], row["x_fin"], None]
            y_vals += [row["y_plot"], row["y_plot"], None]

            txt = (
                f"Fold: {int(row['fold']) + 1}<br>"
                f"Tipo: {nombres[tipo]}<br>"
                f"Segmento: {int(row['ventana_id'])}<br>"
                f"Inicio: {int(row['x_inicio'])}<br>"
                f"Fin: {int(row['x_fin'])}<br>"
                f"Duración: {int(row['duracion'])}"
            )

            text_vals += [txt, txt, None]

        fig.add_trace(
            go.Scattergl(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=colores[tipo], width=2),
                name=nombres[tipo],
                legendgroup=tipo,
                showlegend=bool(fold == folds[0]),
                text=text_vals,
                hovertemplate="%{text}<extra></extra>"
            )
        )

# ============================================================
# Layout
# ============================================================

fig.update_layout(
    title=(
        f"Group K-Fold - timeseries-cv | Pliegues reales | "
        f"ventana de entrada = {num_inputs} días"
    ),
    xaxis_title="Índice temporal real dentro de train/validación interna",
    yaxis_title="Train and validation set",
    template="plotly_white",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=70, r=30, t=90, b=60),
    height=750,
    width=1350
)

fig.update_xaxes(
    range=[-1, n_total],
    showgrid=True
)

fig.update_yaxes(
    range=[len(folds) + 1.0, 0.0],
    tickmode="array",
    tickvals=[fold + 1 for fold in folds],
    ticktext=[str(fold + 1) for fold in folds],
    showgrid=False
)

# ============================================================
# Mostrar y guardar
# ============================================================

fig.show()

fig.write_html(
    str(archivo_html),
    include_plotlyjs="cdn"
)

print("HTML guardado en:")
print(archivo_html)