import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_DATOS = "data/Niveles_imputados_completo.csv"

SALIDA_DATOS_IMPUTADOS = "data/niveles_imputados_iterativo.csv"
SALIDA_METRICAS = "data/metricas_imputacion_iterativa.csv"
SALIDA_DETALLE = "data/detalle_validacion_imputacion_iterativa.csv"

ESTACIONES = [
    "Calamar",
    "Achi",
    "ElBanco",
    "SaladoBlanco",
    "PuertoBerrio",
    "Barrancabermeja",
]

NOMBRES_ESTACIONES = {
    "Calamar": "Calamar",
    "Achi": "Achí",
    "ElBanco": "El Banco",
    "SaladoBlanco": "Salado Blanco",
    "PuertoBerrio": "Puerto Berrío",
    "Barrancabermeja": "Barrancabermeja",
}

# Orden real del procedimiento manual
ORDEN_IMPUTACION = [
    "Calamar",
    "Achi",
    "ElBanco",
    "SaladoBlanco",
    "PuertoBerrio",
    "Barrancabermeja",
]

PORCENTAJE_SIMULADO = 0.20
SEMILLA = 42


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def leer_csv_robusto(ruta):
    df = pd.read_csv(
        ruta,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )

    df.columns = df.columns.str.strip()

    for col in df.columns:
        if "Fecha" in col:
            df = df.rename(columns={col: "Fecha"})

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    for col in df.columns:
        if col != "Fecha":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Fecha").reset_index(drop=True)

    return df


def mape_seguro(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_real != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100


def ajustar_ols(X, y):
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const).fit()
    return model


def predecir_ols(model, X):
    X_const = sm.add_constant(X, has_constant="add")
    return model.predict(X_const)


# ============================================================
# LECTURA DE DATOS ORIGINALES
# ============================================================

df_base = leer_csv_robusto(RUTA_DATOS)

# Si existen columnas *_original, se usan esas como datos originales.
# Si no existen, se usan las columnas normales.
df_original = pd.DataFrame()
df_original["Fecha"] = df_base["Fecha"]

for est in ESTACIONES:
    col_original = f"{est}_original"

    if col_original in df_base.columns:
        df_original[est] = df_base[col_original]
    elif est in df_base.columns:
        df_original[est] = df_base[est]
    else:
        raise ValueError(f"No se encontró la estación {est} ni {col_original} en el archivo.")

df_original["Mes"] = df_original["Fecha"].dt.month


# ============================================================
# PROMEDIO CLIMATOLÓGICO MENSUAL
# ============================================================

climatologia = (
    df_original
    .groupby("Mes")[ESTACIONES]
    .mean()
)

# Crear copia auxiliar.
# Esta copia empieza con los datos originales, pero los faltantes
# se rellenan inicialmente con el promedio climatológico mensual.
df_aux = df_original[["Fecha", "Mes"] + ESTACIONES].copy()

for est in ESTACIONES:
    for mes in range(1, 13):
        mask_mes = df_aux["Mes"] == mes
        valor_clima = climatologia.loc[mes, est]

        df_aux.loc[
            mask_mes & df_aux[est].isna(),
            est
        ] = valor_clima


# ============================================================
# VALIDACIÓN E IMPUTACIÓN ITERATIVA
# ============================================================

metricas_todas = []
detalle_validacion_todas = []

rng = np.random.default_rng(SEMILLA)

for target in ORDEN_IMPUTACION:

    print(f"\nProcesando estación objetivo: {target}")

    predictors = [
        est for est in ESTACIONES
        if est != target
    ]

    # --------------------------------------------------------
    # A. VALIDACIÓN CON FALTANTES SIMULADOS
    # --------------------------------------------------------

    # Para validar, usamos filas donde el target original sí existe.
    # Las predictoras vienen de df_aux, porque representan el estado
    # auxiliar del procedimiento iterativo: climatología + imputaciones previas.
    df_validacion = pd.DataFrame()
    df_validacion["Fecha"] = df_original["Fecha"]
    df_validacion[target] = df_original[target]

    for pred in predictors:
        df_validacion[pred] = df_aux[pred]

    df_validacion = df_validacion.dropna(subset=[target] + predictors).copy()

    if len(df_validacion) < 20:
        print(f"Saltando validación de {target}: pocos datos completos.")
        continue

    mask_simulado = rng.random(len(df_validacion)) < PORCENTAJE_SIMULADO

    if mask_simulado.sum() < 5:
        print(f"Saltando validación de {target}: pocos faltantes simulados.")
        continue

    df_test = df_validacion.copy()

    y_real = df_test.loc[mask_simulado, target].copy()
    df_test.loc[mask_simulado, target] = np.nan

    df_train = df_test.loc[~mask_simulado].dropna(subset=[target] + predictors).copy()

    X_train = df_train[predictors]
    y_train = df_train[target]

    model_validacion = ajustar_ols(X_train, y_train)

    df_eval = df_test.loc[mask_simulado].dropna(subset=predictors).copy()

    X_eval = df_eval[predictors]
    y_pred = predecir_ols(model_validacion, X_eval)

    y_real_eval = y_real.loc[df_eval.index]

    rmse = np.sqrt(mean_squared_error(y_real_eval, y_pred))
    mae = mean_absolute_error(y_real_eval, y_pred)
    r2 = r2_score(y_real_eval, y_pred)
    mape = mape_seguro(y_real_eval, y_pred)

    metricas_todas.append({
        "Estación": NOMBRES_ESTACIONES.get(target, target),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "MAPE [%]": round(mape, 3),
        "R²": round(r2, 3),
        "Filas evaluadas": int(len(y_real_eval)),
        "Faltantes simulados": int(mask_simulado.sum()),
        "Predictoras usadas": ", ".join(
            NOMBRES_ESTACIONES.get(p, p) for p in predictors
        )
    })

    detalle_tmp = pd.DataFrame({
        "Estación": NOMBRES_ESTACIONES.get(target, target),
        "Fecha": df_eval["Fecha"].values,
        "Valor real": y_real_eval.values,
        "Valor predicho": y_pred.values,
    })

    detalle_validacion_todas.append(detalle_tmp)

    print(
        f"{target}: RMSE={rmse:.3f}, MAE={mae:.3f}, "
        f"MAPE={mape:.3f} %, R²={r2:.3f}, n={len(y_real_eval)}"
    )

    # --------------------------------------------------------
    # B. IMPUTACIÓN REAL DE LA ESTACIÓN OBJETIVO
    # --------------------------------------------------------

    # Entrenar modelo final con datos originales observados del target.
    df_modelo_final = pd.DataFrame()
    df_modelo_final[target] = df_original[target]

    for pred in predictors:
        df_modelo_final[pred] = df_aux[pred]

    df_modelo_final = df_modelo_final.dropna(subset=[target] + predictors).copy()

    X_final = df_modelo_final[predictors]
    y_final = df_modelo_final[target]

    model_final = ajustar_ols(X_final, y_final)

    # Identificar faltantes reales del target en la serie original.
    mask_faltantes_reales = df_original[target].isna()

    df_pred_real = pd.DataFrame()

    for pred in predictors:
        df_pred_real[pred] = df_aux.loc[mask_faltantes_reales, pred]

    df_pred_real = df_pred_real.dropna(subset=predictors).copy()

    if not df_pred_real.empty:
        pred_real = predecir_ols(model_final, df_pred_real)

        # Reemplazar en df_aux los valores faltantes reales del target
        # por los valores estimados por regresión.
        df_aux.loc[df_pred_real.index, target] = pred_real

    print(f"Faltantes reales imputados en {target}: {len(df_pred_real)}")


# ============================================================
# EXPORTAR RESULTADOS
# ============================================================

metricas_imputacion = pd.DataFrame(metricas_todas)

if detalle_validacion_todas:
    detalle_validacion_imputacion = pd.concat(
        detalle_validacion_todas,
        ignore_index=True
    )
else:
    detalle_validacion_imputacion = pd.DataFrame(
        columns=["Estación", "Fecha", "Valor real", "Valor predicho"]
    )

# Dataset final imputado
df_salida = df_aux[["Fecha"] + ESTACIONES].copy()

# Agregar columnas originales para referencia
for est in ESTACIONES:
    df_salida[f"{est}_original"] = df_original[est]

os.makedirs("data", exist_ok=True)

df_salida.to_csv(
    SALIDA_DATOS_IMPUTADOS,
    index=False,
    encoding="utf-8-sig"
)

metricas_imputacion.to_csv(
    SALIDA_METRICAS,
    index=False,
    encoding="utf-8-sig"
)

detalle_validacion_imputacion.to_csv(
    SALIDA_DETALLE,
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivos exportados:")
print(SALIDA_DATOS_IMPUTADOS)
print(SALIDA_METRICAS)
print(SALIDA_DETALLE)