# ============================================================
# XARIMA / SARIMAX H10 + timeseries-cv GroupKFold (train/validation)
# Predicción directa de 10 días de nivel en Calamar
#
# Metodología igual a los modelos anteriores:
# - Test externo final: últimos 10 registros del CSV.
# - Validación cruzada interna: split_train_val_groupKFold.
# - NO se usa split_train_val_test_groupKFold.
# - Selección principal: menor MAE de validación H10.
# - BDS como diagnóstico/filtro de residuos de validación.
# - Métricas acumuladas en test externo: h = 1, 5 y 10.
# - Resultados en carpeta propia: Resultados\XARIMA.
#
# Nota importante:
# XARIMA/SARIMAX NO usa Pipeline ni StandardScaler.
# La razón es que ARIMA/SARIMA modela directamente la estructura temporal
# de la serie univariada, no una matriz supervisada X con features escaladas.
# ============================================================

import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsxv.splitTrainVal import split_train_val_groupKFold

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    r"C:\Users\Victus\Documents\Academico\MCT\Semestre2\MachineLearning\PF\src_Entregable2\Modelos\ModelosCorregidos\Resultados\XARIMA"
)
carpeta_salida.mkdir(parents=True, exist_ok=True)

# Archivos tabulares
archivo_resultados = carpeta_salida / "resultados_xarima_h10_timeseries_cv_bds.csv"
archivo_resumen = carpeta_salida / "resumen_xarima_h10_timeseries_cv_bds.csv"
archivo_test_final = carpeta_salida / "test_final_externo_xarima_h10_calamar.csv"
archivo_metricas_horizontes = carpeta_salida / "metricas_horizontes_1_5_10_xarima_h10.csv"

# Figuras
archivo_figura_test = carpeta_salida / "xarima_h10_test_final_externo.png"
archivo_acf_residuos = carpeta_salida / "acf_residuos_xarima_h10_test_externo.png"
archivo_hist_residuos = carpeta_salida / "histograma_residuos_xarima_h10_test_externo.png"

# Modelo y metadata
archivo_modelo = carpeta_salida / "modelo_xarima_h10_calamar.joblib"
archivo_metadata = carpeta_salida / "metadata_modelo_xarima_h10_calamar.json"

# Horizonte directo de predicción
numOutputs = 10
numJumps = 1
H_test = 10
horizontes_eval = [1, 5, 10]

# Para mantener la misma lógica de división temporal que los modelos
# supervisados, se usa split_train_val_groupKFold con estas ventanas.
# En XARIMA estas ventanas NO son features del modelo. Solo definen
# los orígenes de validación temporal.
numInputs_cv_list = [30, 60]

# BDS
alpha_bds = 0.05

# Transformación logarítmica
# Se deja desactivada para mantener la comparación directa en escala original.
usar_log = False

# Grid XARIMA/SARIMAX.
# seasonal_order=(0,0,0,0) equivale a ARIMA no estacional dentro de SARIMAX.
# No se usan AIC/BIC/HQIC para seleccionar, solo MAE/MSE de validación + BDS.
param_grid = {
    "order": [
        (1, 0, 0),
        (2, 0, 0),
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
    ],
    "seasonal_order": [
        (0, 0, 0, 0),
    ],
    "trend": ["n", "c", "t"]
}

maxiter_fit = 200


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


def transformar_serie(y, log=False):
    y = np.asarray(y).astype(float)
    if log:
        if np.any(y <= 0):
            raise ValueError("No se puede usar log porque la serie tiene valores <= 0.")
        return np.log(y)
    return y


def invertir_transformacion(y, log=False):
    y = np.asarray(y).astype(float)
    if log:
        return np.exp(y)
    return y


def ajustar_xarima(y_train, order, seasonal_order, trend):
    """
    Ajusta SARIMAX como XARIMA/SARIMA univariado.
    """
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(
        disp=False,
        maxiter=maxiter_fit
    )

    return fit


def pronosticar_h10_desde_origen(
    sequence,
    forecast_start_idx,
    order,
    seasonal_order,
    trend,
    h=10,
    log=False
):
    """
    Ajusta el modelo usando solamente la información disponible antes del
    horizonte de validación/test y pronostica h pasos.

    forecast_start_idx:
        Índice del primer día que se quiere pronosticar.
        El ajuste usa sequence[:forecast_start_idx].
    """
    y_train = sequence[:forecast_start_idx].astype(float)
    y_train_t = transformar_serie(y_train, log=log)

    fit = ajustar_xarima(
        y_train=y_train_t,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend
    )

    forecast_t = fit.forecast(steps=h)
    forecast = invertir_transformacion(forecast_t, log=log)

    return np.asarray(forecast).astype(float), fit


def obtener_primer_horizonte_validacion(ycv_fold, h=10):
    """
    Toma el primer bloque ycv de validación generado por timeseries-cv.
    Ese bloque contiene los h índices del horizonte de validación.
    """
    arr = np.asarray(ycv_fold)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    idx = arr[0].astype(int)

    return idx[:h]


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

if len(df) <= H_test + max(numInputs_cv_list) + numOutputs:
    raise ValueError(
        "La serie no tiene suficientes datos para separar test externo "
        "y construir la validación."
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
# 5. Validación cruzada temporal con XARIMA/SARIMAX
# ============================================================

resultados = []
n_combinaciones = len(list(ParameterGrid(param_grid)))
n_folds_esperados = 5
n_modelos_aprox = len(numInputs_cv_list) * n_combinaciones * n_folds_esperados

print("\n" + "=" * 80)
print("CONFIGURACIÓN DE BÚSQUEDA XARIMA")
print("=" * 80)
print("Ventanas usadas para definir pliegues:", numInputs_cv_list)
print("Grid XARIMA:", param_grid)
print("Horizonte forecast:", numOutputs)
print("Modelos aproximados en validación:", n_modelos_aprox)

for numInputs_cv in numInputs_cv_list:

    print("\n" + "-" * 80)
    print(f"Pliegues definidos con ventana de referencia: {numInputs_cv} días")
    print("-" * 80)

    # En XARIMA usamos timeseries-cv sobre índices para mantener la misma
    # estructura temporal de validación. El modelo se ajusta sobre la serie
    # univariada hasta el origen de pronóstico.
    indices = np.arange(len(sequence_trainval), dtype=float)

    X_idx, y_idx, Xcv_idx, ycv_idx = split_train_val_groupKFold(
        indices,
        numInputs_cv,
        numOutputs,
        numJumps
    )

    print("Número de folds generados:", len(X_idx))

    for params in ParameterGrid(param_grid):
        order = params["order"]
        seasonal_order = params["seasonal_order"]
        trend = params["trend"]

        mae_val_folds = []
        mse_val_folds = []
        # En XARIMA cada fold pronostica solo H10, por lo que cada fold
        # genera 10 residuos. El BDS requiere más observaciones; por eso
        # se acumulan los residuos de todos los folds de la misma configuración
        # y se calcula un único BDS sobre esos residuos de validación.
        residuos_val_todos = []
        converged_folds = []
        folds_ok = 0
        folds_fail = 0

        for fold in sorted(X_idx.keys()):
            try:
                idx_val = obtener_primer_horizonte_validacion(
                    ycv_fold=ycv_idx[fold],
                    h=numOutputs
                )

                val_start = int(idx_val[0])
                val_end = int(idx_val[-1]) + 1

                y_val_real = sequence_trainval[val_start:val_end]

                # Pronóstico H10 desde el origen val_start.
                # No se usan datos futuros del horizonte y_val_real.
                y_val_pred, fit = pronosticar_h10_desde_origen(
                    sequence=sequence_trainval,
                    forecast_start_idx=val_start,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    h=numOutputs,
                    log=usar_log
                )

                residuos_val = y_val_real - y_val_pred

                mae_val, mse_val = calcular_metricas(y_val_real, y_val_pred)
                mae_val_folds.append(mae_val)
                mse_val_folds.append(mse_val)
                residuos_val_todos.extend(np.asarray(residuos_val).astype(float).ravel().tolist())
                converged_folds.append(bool(fit.mle_retvals.get("converged", False)))

                folds_ok += 1

            except Exception as e:
                folds_fail += 1
                print(
                    f"Fallo fold={fold}, order={order}, seasonal_order={seasonal_order}, "
                    f"trend={trend}. Error: {e}"
                )
                continue

        if folds_ok == 0:
            continue

        # BDS agregado por configuración:
        # 5 folds x 10 días = 50 residuos aproximadamente.
        # Esto evita que el BDS quede vacío por tener solo 10 residuos por fold.
        pvalue_bds_total, pasa_bds_total = calcular_bds_pvalue(
            residuos_val_todos,
            max_dim=2,
            alpha=alpha_bds
        )

        fila = {
            "numInputs_cv": numInputs_cv,
            "ventana_cv": f"{numInputs_cv} días",
            "numOutputs": numOutputs,
            "numJumps": numJumps,
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "trend": trend,
            "usar_log": usar_log,
            "MAE_val_h10_mean": np.mean(mae_val_folds),
            "MAE_val_h10_std": np.std(mae_val_folds),
            "MSE_val_h10_mean": np.mean(mse_val_folds),
            "MSE_val_h10_std": np.std(mse_val_folds),
            "converged_folds": int(np.sum(converged_folds)),
            "folds_ok": folds_ok,
            "folds_fail": folds_fail,
            "BDS_pvalue_mean": pvalue_bds_total,
            "BDS_pvalue_min": pvalue_bds_total,
            "BDS_folds_pass": int(pasa_bds_total),
            "BDS_all_folds_pass": bool(pasa_bds_total),
            "BDS_any_fold_pass": bool(pasa_bds_total),
            "BDS_n_residuos": int(len(residuos_val_todos)),
            "BDS_nota": "BDS calculado sobre residuos agregados de todos los folds de validación"
        }

        resultados.append(fila)

    print(f"Pliegues con ventana de referencia {numInputs_cv} días terminados.")


# ============================================================
# 6. Guardar resultados de validación
# ============================================================

resultados_df = pd.DataFrame(resultados)

if resultados_df.empty:
    raise RuntimeError(
        "No se ajustó ningún modelo XARIMA correctamente. "
        "Revisa el grid de order/seasonal_order/trend."
    )

resultados_df.to_csv(archivo_resultados, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("VALIDACIÓN XARIMA TERMINADA")
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
print("MEJORES CONFIGURACIONES XARIMA")
print("=" * 80)
print(resumen_seleccion[[
    "criterio",
    "criterio_bds_usado",
    "numInputs_cv",
    "ventana_cv",
    "order",
    "seasonal_order",
    "trend",
    "MAE_val_h10_mean",
    "MSE_val_h10_mean",
    "BDS_pvalue_min",
    "BDS_folds_pass",
    "BDS_all_folds_pass"
]])

modelo_final_row = mejor_mae.copy()

best_numInputs_cv = int(modelo_final_row["numInputs_cv"])
best_order = eval(modelo_final_row["order"])
best_seasonal_order = eval(modelo_final_row["seasonal_order"])
best_trend = str(modelo_final_row["trend"])

print("\n" + "=" * 80)
print("MODELO FINAL XARIMA SELECCIONADO")
print("=" * 80)
print(modelo_final_row[[
    "numInputs_cv",
    "ventana_cv",
    "order",
    "seasonal_order",
    "trend",
    "MAE_val_h10_mean",
    "MSE_val_h10_mean",
    "BDS_pvalue_min",
    "BDS_folds_pass",
    "BDS_all_folds_pass"
]])


# ============================================================
# 8. Entrenar modelo final sin incluir test externo
# ============================================================

y_trainval_t = transformar_serie(sequence_trainval, log=usar_log)

modelo_final_fit = ajustar_xarima(
    y_train=y_trainval_t,
    order=best_order,
    seasonal_order=best_seasonal_order,
    trend=best_trend
)


# ============================================================
# 9. Evaluar test externo final: últimos 10 registros
# ============================================================

y_test_ext = df_test_ext["Calamar"].values.astype(float)
fechas_test_ext = df_test_ext["Fecha"].values

y_test_ext_pred_t = modelo_final_fit.forecast(steps=numOutputs)
y_test_ext_pred = invertir_transformacion(y_test_ext_pred_t, log=usar_log)
y_test_ext_pred = np.asarray(y_test_ext_pred).astype(float).ravel()

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
plt.title("Histograma de residuos - XARIMA H10 en test externo")
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
    ax.set_title("ACF de residuos - XARIMA H10 en test externo")
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
    f"XARIMA/SARIMAX - Test externo final H10 | "
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
    "order": best_order,
    "seasonal_order": best_seasonal_order,
    "trend": best_trend,
    "usar_log": usar_log,
    "numInputs_cv": best_numInputs_cv
}

paquete_modelo = {
    "model_fit": modelo_final_fit,
    "target": "Calamar",
    "numOutputs": numOutputs,
    "numJumps": numJumps,
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
    "numOutputs": numOutputs,
    "numJumps": numJumps,
    "horizontes_eval_test": horizontes_eval,
    "numInputs_cv_list": numInputs_cv_list,
    "best_params": best_params,
    "modelo": "SARIMAX usado como XARIMA/SARIMA univariado",
    "criterio_final": (
        "Menor MAE de validación acumulado sobre H10, usando BDS como "
        "criterio de diagnóstico/filtro. En XARIMA el BDS se calcula sobre "
        "los residuos agregados de todos los folds de validación, porque cada fold "
        "solo tiene 10 residuos. El modelo final fue entrenado sin incluir los últimos "
        "10 registros, reservados como test externo final."
    ),
    "nota_metodologica": (
        "A diferencia de SVR/Ridge/Lasso, XARIMA no usa ventanas como features "
        "ni requiere StandardScaler. Las ventanas numInputs_cv solo se usaron "
        "para definir los orígenes de validación con split_train_val_groupKFold."
    ),
    "validacion_cruzada": "split_train_val_groupKFold de timeseries-cv sobre índices temporales",
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
