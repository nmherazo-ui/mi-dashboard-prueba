# ============================================================
# Ridge MultiOutput H10 + timeseries-cv GroupKFold (train/validation)
# Predicción directa de 10 días de nivel en Calamar
#
# Metodología:
# - Test externo final: últimos 10 registros del CSV.
# - Validación cruzada interna: split_train_val_groupKFold.
# - NO se usa split_train_val_test_groupKFold.
# - Modelo: Pipeline(StandardScaler + Ridge).
# - Ridge acepta salida múltiple directamente, por eso NO necesita
#   MultiOutputRegressor.
# - Salida: 10 días predichos de golpe, no recursivo.
# - Ventanas de entrada: 30 y 60 días.
# - Selección: menor MAE de validación H10, usando BDS como filtro/diagnóstico.
# ============================================================

import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsxv.splitTrainVal import split_train_val_groupKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

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
archivo_resultados = carpeta_salida / "resultados_ridge_multioutput_h10_timeseries_cv_bds.csv"
archivo_resumen = carpeta_salida / "resumen_ridge_multioutput_h10_timeseries_cv_bds.csv"
archivo_test_final = carpeta_salida / "test_final_externo_ridge_multioutput_h10_calamar.csv"
archivo_metricas_horizontes = carpeta_salida / "metricas_horizontes_1_5_10_ridge_multioutput_h10.csv"

# Figuras
archivo_figura_test = carpeta_salida / "ridge_multioutput_h10_test_final_externo.png"
archivo_acf_residuos = carpeta_salida / "acf_residuos_ridge_multioutput_h10_test_externo.png"
archivo_hist_residuos = carpeta_salida / "histograma_residuos_ridge_multioutput_h10_test_externo.png"

# Modelo y metadata
archivo_modelo = carpeta_salida / "modelo_ridge_multioutput_h10_calamar.joblib"
archivo_metadata = carpeta_salida / "metadata_modelo_ridge_multioutput_h10_calamar.json"

# Ventanas de entrada
numInputs_list = [30, 60]

# Horizonte directo de predicción
numOutputs = 10
numJumps = 1
H_test = 10
horizontes_eval = [1, 5, 10]

# BDS
alpha_bds = 0.05

# Grid de Ridge
# Ridge es rápido, por eso se puede evaluar un rango pequeño de alpha.
param_grid = {
    "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
}


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
    - Ridge se entrena sobre la salida múltiple H10.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
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
# 5. Validación cruzada temporal con Ridge MultiOutput
# ============================================================

resultados = []
n_combinaciones = len(list(ParameterGrid(param_grid)))
n_folds_esperados = 5
n_pipelines = len(numInputs_list) * n_combinaciones * n_folds_esperados

print("\n" + "=" * 80)
print("CONFIGURACIÓN DE BÚSQUEDA")
print("=" * 80)
print("Ventanas evaluadas:", numInputs_list)
print("Grid Ridge:", param_grid)
print("Horizonte multioutput:", numOutputs)
print("Pipelines aproximados en validación:", n_pipelines)

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

    for params in ParameterGrid(param_grid):
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
            model.set_params(**params)
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
            "alpha": params["ridge__alpha"],
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
# 6. Guardar resultados de validación
# ============================================================

resultados_df = pd.DataFrame(resultados)
resultados_df.to_csv(archivo_resultados, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("VALIDACIÓN TERMINADA")
print("=" * 80)
print("Resultados completos guardados en:", archivo_resultados)


# ============================================================
# 7. Selección del mejor modelo con MAE H10 + BDS
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
    "alpha",
    "MAE_val_h10_mean",
    "MSE_val_h10_mean",
    "BDS_pvalue_min",
    "BDS_folds_pass",
    "BDS_all_folds_pass"
]])

modelo_final_row = mejor_mae.copy()
best_numInputs = int(modelo_final_row["numInputs"])
best_alpha = float(modelo_final_row["alpha"])

print("\n" + "=" * 80)
print("MODELO FINAL SELECCIONADO")
print("=" * 80)
print(modelo_final_row[[
    "numInputs",
    "ventana",
    "alpha",
    "MAE_val_h10_mean",
    "MSE_val_h10_mean",
    "BDS_pvalue_min",
    "BDS_folds_pass",
    "BDS_all_folds_pass"
]])


# ============================================================
# 8. Entrenar modelo final sin incluir test externo
# ============================================================

X_trainval_all, y_trainval_all = crear_ventanas_supervisadas(
    sequence_trainval,
    num_inputs=best_numInputs,
    num_outputs=numOutputs,
    num_jumps=numJumps
)

modelo_final = crear_pipeline()
modelo_final.set_params(ridge__alpha=best_alpha)
modelo_final.fit(X_trainval_all, y_trainval_all)


# ============================================================
# 9. Evaluar test externo final: últimos 10 registros
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
# 10. Diagnóstico de residuos en test externo
# ============================================================

plt.figure(figsize=(9, 5))
plt.hist(df_test_final["Residuo"], bins=min(10, len(df_test_final)), edgecolor="black", alpha=0.75)
plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Residuo")
plt.ylabel("Frecuencia")
plt.title("Histograma de residuos - Ridge MultiOutput H10 en test externo")
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
    ax.set_title("ACF de residuos - Ridge MultiOutput H10 en test externo")
    ax.set_xlabel("Rezago")
    ax.set_ylabel("Autocorrelación")
    plt.tight_layout()
    plt.savefig(archivo_acf_residuos, dpi=200)
    plt.show()
    plt.close(fig)
else:
    print("No hay suficientes residuos para graficar la ACF.")


# ============================================================
# 11. Figura test externo: 10 predicciones vs 10 reales
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
    f"Ridge MultiOutput - Test externo final H10 | "
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
# 12. Guardar modelo y metadata
# ============================================================

best_params = {
    "ridge__alpha": best_alpha
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
    "modelo": "Pipeline(StandardScaler + Ridge)",
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
    "archivo_figura_test": str(archivo_figura_test),
    "archivo_acf_residuos": str(archivo_acf_residuos),
    "archivo_hist_residuos": str(archivo_hist_residuos),
}

with open(archivo_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)


# ============================================================
# 13. Mensaje final
# ============================================================

print("\n" + "=" * 80)
print("ARCHIVOS GUARDADOS")
print("=" * 80)
print("Resultados completos :", archivo_resultados)
print("Resumen selección    :", archivo_resumen)
print("Test externo final   :", archivo_test_final)
print("Métricas horizontes  :", archivo_metricas_horizontes)
print("Modelo joblib        :", archivo_modelo)
print("Metadata JSON        :", archivo_metadata)
print("Figura test externo  :", archivo_figura_test)
print("ACF residuos         :", archivo_acf_residuos)
print("Histograma residuos  :", archivo_hist_residuos)
