# ============================================================
# SVR MultiOutput H10 + timeseries-cv GroupKFold (train/validation)
# Predicción directa de 10 días de nivel en Calamar
#
# Cambios metodológicos solicitados:
# - Test externo final: últimos 10 registros del CSV.
# - Validación cruzada interna: split_train_val_groupKFold
#   (NO se usa split_train_val_test_groupKFold).
# - Modelo: Pipeline(StandardScaler + MultiOutputRegressor(SVR RBF)).
# - Salida: 10 días predichos de golpe, no recursivo.
# - Ventanas de entrada: 30 y 60 días.
# - Hiperparámetros fijos: C=60, epsilon=0.01, gamma=0.03.
# - Pliegues REALES de la mejor ventana en Plotly interactivo,
#   construidos con la misma función split_train_val_groupKFold
#   aplicada a una serie de índices posicionales de train/validación.
# ============================================================

import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tsxv.splitTrainVal import split_train_val_groupKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.stattools import bds
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")


# ============================================================
# 1. Configuración general
# ============================================================

ruta = Path(
    r"C:\Users\Victus\Documents\Academico\MCT\Semestre2\MachineLearning\PF\DatosNivel\Niveles_imputados.csv"
)

carpeta_salida = Path(
    r"C:\Users\Victus\Documents\Academico\MCT\Semestre2\MachineLearning\PF\src_Entregable2\Modelos\ModelosCorregidos\Resultados"
)
carpeta_salida.mkdir(parents=True, exist_ok=True)

# Archivos tabulares
archivo_resultados = carpeta_salida / "resultados_svr_multioutput_h10_fijo_timeseries_cv_bds.csv"
archivo_resumen = carpeta_salida / "resumen_svr_multioutput_h10_fijo_timeseries_cv_bds.csv"
archivo_test_final = carpeta_salida / "test_final_externo_svr_multioutput_h10_calamar.csv"
archivo_metricas_horizontes = carpeta_salida / "metricas_horizontes_1_5_10_svr_multioutput_h10.csv"
archivo_plotly_pliegues_todas = carpeta_salida / "plotly_pliegues_groupkfold_svr_multioutput_h10.csv"
archivo_plotly_pliegues_mejor = carpeta_salida / "plotly_pliegues_reales_mejor_ventana_svr_multioutput_h10.csv"
archivo_info_pliegues = carpeta_salida / "info_pliegues_groupkfold_svr_multioutput_h10.csv"

# Archivos interactivos y figuras
archivo_html_pliegues_mejor = carpeta_salida / "plotly_pliegues_reales_mejor_ventana_svr_multioutput_h10.html"
archivo_figura_test = carpeta_salida / "svr_multioutput_h10_test_final_externo.png"
archivo_acf_residuos = carpeta_salida / "acf_residuos_svr_multioutput_h10_test_externo.png"
archivo_hist_residuos = carpeta_salida / "histograma_residuos_svr_multioutput_h10_test_externo.png"

# Modelo y metadata
archivo_modelo = carpeta_salida / "modelo_svr_multioutput_h10_calamar.joblib"
archivo_metadata = carpeta_salida / "metadata_modelo_svr_multioutput_h10_calamar.json"

# Ventanas de entrada solicitadas
numInputs_list = [30, 60]

# Horizonte directo de predicción
numOutputs = 10
numJumps = 1
H_test = 10
horizontes_eval = [1, 5, 10]

# BDS
alpha_bds = 0.05

# Hiperparámetros fijos solicitados
C_fijo = 60
epsilon_fijo = 0.01
gamma_fijo = 0.03
kernel_fijo = "rbf"


# ============================================================
# 2. Funciones auxiliares
# ============================================================

def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    return mae, mse


def calcular_metricas_horizontes(y_real, y_pred, horizontes):
    """
    Calcula métricas acumuladas hasta cada horizonte.
    h=5 evalúa los días 1, 2, 3, 4 y 5.
    """
    y_real = np.asarray(y_real).astype(float).ravel()
    y_pred = np.asarray(y_pred).astype(float).ravel()

    resultados = []
    for h in horizontes:
        y_h = y_real[:h]
        yhat_h = y_pred[:h]

        mae = mean_absolute_error(y_h, yhat_h)
        mse = mean_squared_error(y_h, yhat_h)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_h, yhat_h) if len(y_h) >= 2 else np.nan

        mask = y_h != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_h[mask] - yhat_h[mask]) / y_h[mask])) * 100
        else:
            mape = np.nan

        resultados.append({
            "horizonte": h,
            "n_datos": len(y_h),
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAPE_pct": mape
        })

    return resultados


def calcular_bds_pvalue(residuos, max_dim=2, alpha=0.05):
    """
    Aplica BDS sobre residuos aplanados.
    En salida multioutput, residuos tiene forma n_muestras x horizonte.
    """
    residuos = np.asarray(residuos).astype(float).ravel()
    residuos = residuos[~np.isnan(residuos)]

    if len(residuos) < 20:
        return np.nan, False

    try:
        _, pvalue = bds(residuos, max_dim=max_dim)
        pvalue = np.asarray(pvalue).astype(float)
        pvalue_min = np.nanmin(pvalue)
        pasa_bds = bool(pvalue_min >= alpha)
        return pvalue_min, pasa_bds
    except Exception:
        return np.nan, False


def crear_pipeline():
    """
    Pipeline sin fuga de datos:
    - StandardScaler se ajusta solo con el train de cada fold.
    - MultiOutputRegressor ajusta un SVR independiente para cada horizonte.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", MultiOutputRegressor(
            SVR(
                kernel=kernel_fijo,
                C=C_fijo,
                epsilon=epsilon_fijo,
                gamma=gamma_fijo
            ),
            n_jobs=-1
        ))
    ])
    return pipe


def crear_ventanas_supervisadas(sequence, num_inputs, num_outputs=10, num_jumps=1):
    """
    Crea ventanas supervisadas multioutput.
    X: últimos num_inputs valores.
    y: próximos num_outputs valores.
    """
    sequence = np.asarray(sequence).astype(float)
    X_all = []
    y_all = []

    max_start = len(sequence) - num_inputs - num_outputs + 1

    for start in range(0, max_start, num_jumps):
        end_input = start + num_inputs
        end_output = end_input + num_outputs
        X_all.append(sequence[start:end_input])
        y_all.append(sequence[end_input:end_output])

    return np.asarray(X_all), np.asarray(y_all)


def generar_segmentos_pliegues_reales(n_total, num_inputs, num_outputs, num_jumps):
    """
    Genera los segmentos REALES de los pliegues usando la misma función
    split_train_val_groupKFold aplicada a una serie de índices posicionales
    de longitud n_total.

    Esto representa los pliegues reales dentro de train/validación interna.
    """
    serie_indices = np.arange(n_total, dtype=float)

    X_idx, y_idx, Xcv_idx, ycv_idx = split_train_val_groupKFold(
        serie_indices,
        num_inputs,
        num_outputs,
        num_jumps
    )

    registros_segmentos = []
    registros_info = []

    datasets = [
        ("X", "X (Train input)", X_idx, "blue", -0.14),
        ("y", "y (Train output)", y_idx, "lightblue", -0.04),
        ("Xcv", "Xcv (Cross-validation input)", Xcv_idx, "red", 0.10),
        ("ycv", "ycv (Cross-validation output)", ycv_idx, "tomato", 0.20),
    ]

    for fold in sorted(X_idx.keys()):
        registros_info.append({
            "numInputs": int(num_inputs),
            "fold": int(fold),
            "n_train": int(len(X_idx[fold])),
            "n_val": int(len(Xcv_idx[fold])),
            "inicio_train_input": int(np.nanmin(X_idx[fold])) if len(X_idx[fold]) else np.nan,
            "fin_train_output": int(np.nanmax(y_idx[fold])) if len(y_idx[fold]) else np.nan,
            "inicio_val_input": int(np.nanmin(Xcv_idx[fold])) if len(Xcv_idx[fold]) else np.nan,
            "fin_val_output": int(np.nanmax(ycv_idx[fold])) if len(ycv_idx[fold]) else np.nan,
        })

        for tipo, etiqueta, diccionario, color, offset_y in datasets:
            arr = diccionario[fold]
            if arr is None or len(arr) == 0:
                continue

            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            for i, fila in enumerate(arr):
                registros_segmentos.append({
                    "numInputs": int(num_inputs),
                    "fold": int(fold),
                    "tipo_segmento": tipo,
                    "conjunto": etiqueta,
                    "ventana_id": int(i),
                    "x_inicio": int(np.nanmin(fila)),
                    "x_fin": int(np.nanmax(fila)),
                    "duracion": int(np.nanmax(fila) - np.nanmin(fila) + 1),
                    "y_base": int(fold) + 1,
                    "offset_y": float(offset_y),
                    "y_plot": float(int(fold) + 1 + offset_y),
                    "color_sugerido": color,
                    "etiqueta_y": f"win={num_inputs} | fold={int(fold)}"
                })

    return pd.DataFrame(registros_segmentos), pd.DataFrame(registros_info)


def graficar_pliegues_plotly_reales(df_segmentos, n_total, num_inputs, archivo_html):
    """
    Genera un HTML Plotly INTERACTIVO con los pliegues REALES de la mejor ventana,
    usando todos los índices reales de train/validación interna.
    """
    df_plot = df_segmentos[df_segmentos["numInputs"] == num_inputs].copy()

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

    folds = sorted(df_plot["fold"].unique())
    fig = go.Figure()

    # Línea superior de referencia
    fig.add_trace(go.Scattergl(
        x=[0, n_total - 1],
        y=[0.35, 0.35],
        mode="lines",
        line=dict(color="black", width=3),
        name="Original Time-Series",
        hoverinfo="skip"
    ))

    # Área sombreada de cada fold
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

    # Un trace por fold y por tipo, para que el hover sea más claro.
    orden = ["X", "y", "Xcv", "ycv"]
    for tipo in orden:
        for fold in folds:
            sub = df_plot[(df_plot["tipo_segmento"] == tipo) & (df_plot["fold"] == fold)].copy()
            if sub.empty:
                continue

            x_vals, y_vals, text_vals = [], [], []
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

            fig.add_trace(go.Scattergl(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=colores[tipo], width=2),
                name=nombres[tipo],
                legendgroup=tipo,
                showlegend=bool(fold == folds[0]),
                text=text_vals,
                hovertemplate="%{text}<extra></extra>"
            ))

    fig.update_layout(
        title=(
            f"Group K-Fold - timeseries-cv | Pliegues REALES | "
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

    fig.update_xaxes(range=[-1, n_total], showgrid=True)
    fig.update_yaxes(
        range=[len(folds) + 1.0, 0.0],
        tickmode="array",
        tickvals=[fold + 1 for fold in folds],
        ticktext=[str(fold + 1) for fold in folds],
        showgrid=False
    )

    fig.write_html(str(archivo_html), include_plotlyjs="cdn")
    return fig


# ============================================================
# 3. Cargar datos
# ============================================================

if not ruta.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo de datos en:\n{ruta}\n\n"
        "Ajusta la variable ruta al inicio del script."
    )

df = pd.read_csv(ruta)
df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Fecha", "Calamar"])
df = df.sort_values("Fecha").reset_index(drop=True)

print("=" * 80)
print("DATOS CARGADOS")
print("=" * 80)
print("Dimensión del dataset:", df.shape)
print("Rango temporal:", df["Fecha"].min(), "->", df["Fecha"].max())


# ============================================================
# 4. Separar test externo final: últimos 10 registros
# ============================================================

if len(df) <= H_test + max(numInputs_list) + numOutputs:
    raise ValueError(
        "La serie no tiene suficientes datos para separar test externo "
        "y construir ventanas supervisadas."
    )

df_trainval = df.iloc[:-H_test].copy()
df_test_ext = df.iloc[-H_test:].copy()
sequence_trainval = df_trainval["Calamar"].values.astype(float)

print("\n" + "=" * 80)
print("SEPARACIÓN TEMPORAL")
print("=" * 80)
print("Train/validación interna:", df_trainval["Fecha"].min(), "->", df_trainval["Fecha"].max())
print("Test externo final      :", df_test_ext["Fecha"].min(), "->", df_test_ext["Fecha"].max())
print("Muestras train/val:", len(df_trainval))
print("Muestras test externo:", len(df_test_ext))


# ============================================================
# 5. Construir pliegues REALES usando split_train_val_groupKFold
# ============================================================

segmentos_todos = []
info_todos = []
segmentos_por_ventana = {}

for numInputs in numInputs_list:
    df_seg, df_info = generar_segmentos_pliegues_reales(
        n_total=len(sequence_trainval),
        num_inputs=numInputs,
        num_outputs=numOutputs,
        num_jumps=numJumps
    )
    segmentos_todos.append(df_seg)
    info_todos.append(df_info)
    segmentos_por_ventana[numInputs] = df_seg.copy()

plotly_pliegues_df = pd.concat(segmentos_todos, ignore_index=True)
info_pliegues_df = pd.concat(info_todos, ignore_index=True)
plotly_pliegues_df.to_csv(archivo_plotly_pliegues_todas, index=False, encoding="utf-8-sig")
info_pliegues_df.to_csv(archivo_info_pliegues, index=False, encoding="utf-8-sig")


# ============================================================
# 6. Validación cruzada temporal con SVR MultiOutput
# ============================================================

resultados = []

print("\n" + "=" * 80)
print("CONFIGURACIÓN DE BÚSQUEDA")
print("=" * 80)
print("Ventanas evaluadas:", numInputs_list)
print("Hiperparámetros fijos:", {"C": C_fijo, "epsilon": epsilon_fijo, "gamma": gamma_fijo, "kernel": kernel_fijo})
print("Horizonte multioutput:", numOutputs)

for numInputs in numInputs_list:
    print("\n" + "-" * 80)
    print(f"Ventana de entrada: {numInputs} días")
    print("-" * 80)

    X, y, Xcv, ycv = split_train_val_groupKFold(
        sequence_trainval,
        numInputs,
        numOutputs,
        numJumps
    )

    print("Número de folds generados:", len(X))

    mae_val_folds = []
    mse_val_folds = []
    bds_pvalues = []
    bds_flags = []

    for fold in sorted(X.keys()):
        X_train_fold = np.asarray(X[fold]).astype(float)
        y_train_fold = np.asarray(y[fold]).astype(float)

        X_val_fold = np.asarray(Xcv[fold]).astype(float)
        y_val_fold = np.asarray(ycv[fold]).astype(float)

        model = crear_pipeline()
        model.fit(X_train_fold, y_train_fold)

        y_val_pred = model.predict(X_val_fold)
        residuos_val = y_val_fold - y_val_pred

        mae_val, mse_val = calcular_metricas(y_val_fold.ravel(), y_val_pred.ravel())
        pvalue_bds, pasa_bds = calcular_bds_pvalue(
            residuos_val,
            max_dim=2,
            alpha=alpha_bds
        )

        mae_val_folds.append(mae_val)
        mse_val_folds.append(mse_val)
        bds_pvalues.append(pvalue_bds)
        bds_flags.append(pasa_bds)

    fila = {
        "numInputs": numInputs,
        "ventana": f"{numInputs} días",
        "numOutputs": numOutputs,
        "numJumps": numJumps,
        "C": C_fijo,
        "epsilon": epsilon_fijo,
        "gamma": gamma_fijo,
        "kernel": kernel_fijo,
        "MAE_val_h10_mean": np.mean(mae_val_folds),
        "MAE_val_h10_std": np.std(mae_val_folds),
        "MSE_val_h10_mean": np.mean(mse_val_folds),
        "MSE_val_h10_std": np.std(mse_val_folds),
        "BDS_pvalue_mean": np.nanmean(bds_pvalues),
        "BDS_pvalue_min": np.nanmin(bds_pvalues),
        "BDS_folds_pass": int(np.sum(bds_flags)),
        "BDS_all_folds_pass": bool(np.all(bds_flags)),
        "BDS_any_fold_pass": bool(np.any(bds_flags))
    }

    resultados.append(fila)
    print(f"Ventana {numInputs} días terminada.")


# ============================================================
# 7. Guardar resultados de validación
# ============================================================

resultados_df = pd.DataFrame(resultados)
resultados_df.to_csv(archivo_resultados, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("VALIDACIÓN TERMINADA")
print("=" * 80)
print("Resultados completos guardados en:", archivo_resultados)


# ============================================================
# 8. Selección del mejor modelo con MAE H10 + BDS
# ============================================================

candidatos_bds = resultados_df[resultados_df["BDS_all_folds_pass"] == True].copy()

if len(candidatos_bds) > 0:
    criterio_bds_usado = "BDS_all_folds_pass"
    base_seleccion = candidatos_bds
else:
    criterio_bds_usado = "max_BDS_folds_pass"
    base_seleccion = resultados_df.sort_values(
        ["BDS_folds_pass", "MAE_val_h10_mean", "MSE_val_h10_mean"],
        ascending=[False, True, True]
    )

mejor_mae = (
    base_seleccion
    .sort_values(["MAE_val_h10_mean", "MSE_val_h10_mean"], ascending=True)
    .iloc[0]
)

mejor_mse = (
    base_seleccion
    .sort_values(["MSE_val_h10_mean", "MAE_val_h10_mean"], ascending=True)
    .iloc[0]
)

resumen_seleccion = pd.DataFrame([
    {
        "criterio": "MAE_h10",
        "criterio_bds_usado": criterio_bds_usado,
        **mejor_mae.to_dict()
    },
    {
        "criterio": "MSE_h10",
        "criterio_bds_usado": criterio_bds_usado,
        **mejor_mse.to_dict()
    }
])

resumen_seleccion.to_csv(archivo_resumen, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("MEJORES CONFIGURACIONES")
print("=" * 80)
print(resumen_seleccion[[
    "criterio",
    "criterio_bds_usado",
    "numInputs",
    "ventana",
    "C",
    "epsilon",
    "gamma",
    "MAE_val_h10_mean",
    "MSE_val_h10_mean",
    "BDS_pvalue_min",
    "BDS_folds_pass",
    "BDS_all_folds_pass"
]])

modelo_final_row = mejor_mae.copy()
best_numInputs = int(modelo_final_row["numInputs"])

print("\n" + "=" * 80)
print("MODELO FINAL SELECCIONADO")
print("=" * 80)
print(modelo_final_row[[
    "numInputs",
    "ventana",
    "C",
    "epsilon",
    "gamma",
    "MAE_val_h10_mean",
    "MSE_val_h10_mean",
    "BDS_pvalue_min",
    "BDS_folds_pass",
    "BDS_all_folds_pass"
]])

# Graficar y exportar SOLO los pliegues REALES de la mejor ventana
# usando todos los índices reales de train/validación interna.
df_seg_mejor_real = segmentos_por_ventana[best_numInputs].copy()
df_seg_mejor_real.to_csv(
    archivo_plotly_pliegues_mejor,
    index=False,
    encoding="utf-8-sig"
)

graficar_pliegues_plotly_reales(
    df_segmentos=df_seg_mejor_real,
    n_total=len(sequence_trainval),
    num_inputs=best_numInputs,
    archivo_html=archivo_html_pliegues_mejor
)

print(f"CSV de pliegues REALES de la mejor ventana guardado: {archivo_plotly_pliegues_mejor}")
print(f"HTML interactivo de pliegues REALES guardado: {archivo_html_pliegues_mejor}")


# ============================================================
# 9. Entrenar modelo final sin incluir test externo
# ============================================================

X_trainval_all, y_trainval_all = crear_ventanas_supervisadas(
    sequence_trainval,
    num_inputs=best_numInputs,
    num_outputs=numOutputs,
    num_jumps=numJumps
)

modelo_final = crear_pipeline()
modelo_final.fit(X_trainval_all, y_trainval_all)


# ============================================================
# 10. Evaluar test externo final: últimos 10 registros
# ============================================================

X_test_ext_final = df_trainval["Calamar"].values.astype(float)[-best_numInputs:].reshape(1, -1)
y_test_ext = df_test_ext["Calamar"].values.astype(float)
fechas_test_ext = df_test_ext["Fecha"].values

y_test_ext_pred = modelo_final.predict(X_test_ext_final).ravel()
residuos_test_ext = y_test_ext - y_test_ext_pred

mae_test_ext, mse_test_ext = calcular_metricas(y_test_ext, y_test_ext_pred)

metricas_horizontes = calcular_metricas_horizontes(
    y_real=y_test_ext,
    y_pred=y_test_ext_pred,
    horizontes=horizontes_eval
)

df_metricas_horizontes = pd.DataFrame(metricas_horizontes)
df_metricas_horizontes.to_csv(archivo_metricas_horizontes, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("EVALUACIÓN FINAL EN TEST EXTERNO H10")
print("=" * 80)
print("Periodo test externo:", pd.to_datetime(fechas_test_ext).min(), "->", pd.to_datetime(fechas_test_ext).max())
print(f"MAE test externo H10: {mae_test_ext:.4f}")
print(f"MSE test externo H10: {mse_test_ext:.4f}")
print("\nMétricas por horizonte acumulado:")
print(df_metricas_horizontes)


df_test_final = pd.DataFrame({
    "Fecha": pd.to_datetime(fechas_test_ext),
    "horizonte": np.arange(1, numOutputs + 1),
    "Calamar_real": y_test_ext,
    "Calamar_predicho": y_test_ext_pred,
    "Residuo": residuos_test_ext
})

df_test_final.to_csv(archivo_test_final, index=False, encoding="utf-8-sig")


# ============================================================
# 11. Diagnóstico de residuos en test externo
# ============================================================

plt.figure(figsize=(9, 5))
plt.hist(df_test_final["Residuo"], bins=min(10, len(df_test_final)), edgecolor="black", alpha=0.75)
plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Residuo")
plt.ylabel("Frecuencia")
plt.title("Histograma de residuos - SVR MultiOutput H10 en test externo")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(archivo_hist_residuos, dpi=200)
plt.show()
plt.close()

residuos_acf = df_test_final["Residuo"].dropna()
max_lags = min(9, len(residuos_acf) - 1)

if max_lags >= 1:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(residuos_acf, lags=max_lags, ax=ax)
    ax.set_title("ACF de residuos - SVR MultiOutput H10 en test externo")
    ax.set_xlabel("Rezago")
    ax.set_ylabel("Autocorrelación")
    plt.tight_layout()
    plt.savefig(archivo_acf_residuos, dpi=200)
    plt.show()
    plt.close(fig)
else:
    print("No hay suficientes residuos para graficar la ACF.")


# ============================================================
# 12. Figura test externo: 10 predicciones vs 10 reales
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(
    df_test_final["Fecha"],
    df_test_final["Calamar_real"],
    label="Calamar real",
    linewidth=2.0,
    marker="o"
)
plt.plot(
    df_test_final["Fecha"],
    df_test_final["Calamar_predicho"],
    label="Calamar predicho",
    linewidth=2.0,
    linestyle="--",
    marker="o"
)
plt.xlabel("Fecha")
plt.ylabel("Nivel en Calamar")
plt.title(
    f"SVR MultiOutput - Test externo final H10 | "
    f"MAE={mae_test_ext:.2f}, MSE={mse_test_ext:.2f}"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(archivo_figura_test, dpi=200)
plt.show()
plt.close()
print(f"Figura test externo guardada: {archivo_figura_test}")


# ============================================================
# 13. Guardar modelo y metadata
# ============================================================

best_params = {
    "modelo__estimator__kernel": kernel_fijo,
    "modelo__estimator__C": C_fijo,
    "modelo__estimator__epsilon": epsilon_fijo,
    "modelo__estimator__gamma": gamma_fijo
}

paquete_modelo = {
    "model": modelo_final,
    "target": "Calamar",
    "numInputs": best_numInputs,
    "numOutputs": numOutputs,
    "numJumps": numJumps,
    "feature_description": f"Últimos {best_numInputs} valores diarios de Calamar",
    "best_params": best_params,
    "selection_summary": modelo_final_row.to_dict(),
    "test_externo": {
        "fecha_inicio": str(pd.to_datetime(fechas_test_ext).min()),
        "fecha_fin": str(pd.to_datetime(fechas_test_ext).max()),
        "MAE": float(mae_test_ext),
        "MSE": float(mse_test_ext)
    }
}

joblib.dump(paquete_modelo, archivo_modelo)

metadata = {
    "target": "Calamar",
    "numInputs_seleccionado": best_numInputs,
    "numOutputs": numOutputs,
    "numJumps": numJumps,
    "ventanas_evaluadas": numInputs_list,
    "horizontes_eval_test": horizontes_eval,
    "best_params": best_params,
    "modelo": "Pipeline(StandardScaler + MultiOutputRegressor(SVR RBF))",
    "criterio_final": (
        "Menor MAE de validación acumulado sobre la salida completa H10, "
        "usando BDS como criterio de diagnóstico/filtro. "
        "El modelo final fue entrenado sin incluir los últimos 10 registros, "
        "reservados como test externo final."
    ),
    "validacion_cruzada": "split_train_val_groupKFold de timeseries-cv",
    "fecha_inicio_trainval": str(df_trainval["Fecha"].min()),
    "fecha_fin_trainval": str(df_trainval["Fecha"].max()),
    "fecha_inicio_test_externo": str(df_test_ext["Fecha"].min()),
    "fecha_fin_test_externo": str(df_test_ext["Fecha"].max()),
    "MAE_test_externo": float(mae_test_ext),
    "MSE_test_externo": float(mse_test_ext),
    "archivo_resultados": str(archivo_resultados),
    "archivo_resumen": str(archivo_resumen),
    "archivo_test_final": str(archivo_test_final),
    "archivo_metricas_horizontes": str(archivo_metricas_horizontes),
    "archivo_modelo": str(archivo_modelo),
    "archivo_plotly_pliegues_todas": str(archivo_plotly_pliegues_todas),
    "archivo_plotly_pliegues_mejor": str(archivo_plotly_pliegues_mejor),
    "archivo_info_pliegues": str(archivo_info_pliegues),
    "archivo_html_pliegues_mejor": str(archivo_html_pliegues_mejor),
    "archivo_figura_test": str(archivo_figura_test),
    "archivo_acf_residuos": str(archivo_acf_residuos),
    "archivo_hist_residuos": str(archivo_hist_residuos),
    "nota_pliegues": (
        "El HTML Plotly y el CSV de la mejor ventana contienen los pliegues REALES "
        "de train/validación interna generados con split_train_val_groupKFold. "
        "No incluyen test interno, porque en esta metodología el único test es el "
        "test externo final de 10 días."
    )
}

with open(archivo_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)


# ============================================================
# 14. Mensaje final
# ============================================================

print("\n" + "=" * 80)
print("ARCHIVOS GUARDADOS")
print("=" * 80)
print("Resultados completos                 :", archivo_resultados)
print("Resumen selección                    :", archivo_resumen)
print("Test externo final                   :", archivo_test_final)
print("Métricas horizontes                  :", archivo_metricas_horizontes)
print("Modelo joblib                        :", archivo_modelo)
print("Metadata JSON                        :", archivo_metadata)
print("Figura test externo                  :", archivo_figura_test)
print("ACF residuos                         :", archivo_acf_residuos)
print("Histograma residuos                  :", archivo_hist_residuos)
print("Info pliegues                        :", archivo_info_pliegues)
print("CSV pliegues reales (todas ventanas) :", archivo_plotly_pliegues_todas)
print("CSV pliegues reales (mejor ventana)  :", archivo_plotly_pliegues_mejor)
print("HTML pliegues reales (mejor ventana) :", archivo_html_pliegues_mejor)
