
# app.py
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)
server = app.server  # necesario para Render

# Datos de ejemplo
df = pd.DataFrame({
    "Mes": ["Ene", "Feb", "Mar", "Abr", "May", "Jun"],
    "Ventas": [120, 95, 140, 180, 160, 210],
    "Gastos": [80, 70, 100, 130, 110, 150]
})

# Gráficas
fig_bar = px.bar(df, x="Mes", y="Ventas", title="Ventas por mes", color_discrete_sequence=["#7F77DD"])
fig_line = px.line(df, x="Mes", y=["Ventas", "Gastos"], title="Ventas vs Gastos")

# Layout
app.layout = html.Div([

    html.H1("Mi Dashboard", style={"textAlign": "center", "color": "#7F77DD"}),

    html.Div([
        dcc.Graph(figure=fig_bar, style={"width": "50%"}),
        dcc.Graph(figure=fig_line, style={"width": "50%"})
    ], style={"display": "flex"}),

], style={"fontFamily": "Arial", "padding": "20px"})

if __name__ == "__main__":
    app.run(debug=True)