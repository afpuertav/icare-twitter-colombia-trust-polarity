# -*- coding: utf-8 -*-
"""
s03_datos_geo.py — construccion del conjunto etiquetado para la
clasificacion de origen colombiano. Modulo compartido.

Por que existe este modulo
--------------------------
Los cinco modelos de la etapa 03b tienen que entrenarse y evaluarse sobre
EXACTAMENTE la misma particion, o las metricas no son comparables entre si
y el "criterio final de clasificacion" que pide el asesor no se sostiene.
La particion se calcula una sola vez, se guarda en disco
(intermedios/s03b_particion_geo.parquet) y todos los modelos la releen.

Supervision distante: de donde salen las etiquetas
--------------------------------------------------
No hay anotacion humana de origen geografico. La etiqueta procede del
campo `Location` del perfil, que Twitter entrega ya normalizado a codigo
ISO de pais:

    Location == 'CO'          -> y = 1  (colombiano)
    Location == otro pais     -> y = 0  (no colombiano)
    Location vacia            -> sin etiqueta, es lo que se predice

SESGO QUE HAY QUE DECLARAR EN EL MANUSCRITO. Solo alrededor del 3% de los
perfiles del corpus declaran pais. Quien declara ubicacion no es una
muestra aleatoria de quien no la declara: tiende a ser cuenta mas
establecida, institucional o verificada. El clasificador aprende de esa
subpoblacion y se aplica al 97% restante. Es una limitacion estructural
del diseno, no un defecto corregible con mas datos, y debe aparecer en
Limitations junto con el hecho de que no existe validacion humana del
origen geografico.

Desbalance
----------
Dentro del subconjunto etiquetado, Colombia es minoritaria (~1 de cada
10). Se maneja asi:
  * la particion es ESTRATIFICADA, para que train/val/test conserven la
    proporcion original;
  * el balanceo por submuestreo se aplica SOLO al conjunto de
    entrenamiento, nunca a validacion ni a prueba;
  * las metricas se reportan sobre el conjunto de prueba con su
    distribucion NATURAL, que es la que el modelo encontrara en
    produccion. Por eso se privilegian F1, precision y recall de la clase
    positiva y PR-AUC sobre el accuracy, que con 10:1 es enganoso.

Error corregido respecto de la version 1 del pipeline
-----------------------------------------------------
Los scripts metodoDos*.py balanceaban con `sample(n=min_len,
replace=True)` ANTES de dividir en train/test. Al muestrear con
reemplazo y partir despues, una misma fila podia quedar en entrenamiento
y en evaluacion a la vez: fuga de datos que inflaba las metricas. Aqui se
divide primero y se balancea solo el entrenamiento, sin reemplazo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
import comun


def construir_particion(log, forzar: bool = False) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por tuit del corpus y columnas:
       idx, texto_limpio, etiqueta (1/0/NaN), particion ('train'/'val'/'test'/'sin_etiqueta')
    """
    if cfg.PQ_SPLIT_GEO.exists() and not forzar:
        log.info("Reutilizando particion existente: %s", cfg.PQ_SPLIT_GEO.name)
        return pd.read_parquet(cfg.PQ_SPLIT_GEO)

    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(cfg.PQ_TWITTER_PREP, columns=["texto_limpio", "location"])
    df = df.reset_index(drop=True)
    df["idx"] = np.arange(len(df))
    df["texto_limpio"] = df["texto_limpio"].fillna("").astype(str)

    loc = df["location"].fillna("").str.strip().str.upper()
    etiqueta = pd.Series(np.nan, index=df.index, dtype="float")
    etiqueta[loc == "CO"] = 1.0
    etiqueta[(loc != "") & (loc != "CO")] = 0.0
    df["etiqueta"] = etiqueta

    etiquetados = df[df["etiqueta"].notna()].copy()
    n_pos = int((etiquetados["etiqueta"] == 1).sum())
    n_neg = int((etiquetados["etiqueta"] == 0).sum())
    log.info("Conjunto etiquetado por supervision distante: %s filas "
             "(%s colombianos, %s de otros paises, razon 1:%.1f)",
             f"{len(etiquetados):,}", f"{n_pos:,}", f"{n_neg:,}",
             n_neg / max(n_pos, 1))
    log.info("Sin etiqueta (Location vacia), es lo que se predice: %s (%.1f%% del corpus)",
             f"{int(df['etiqueta'].isna().sum()):,}",
             100 * df["etiqueta"].isna().mean())

    if n_pos < 50 or n_neg < 50:
        log.error("Muy pocas filas etiquetadas para entrenar. Revisar el campo Location.")
        sys.exit(1)

    # train (60) / val (20) / test (20), estratificado
    idx_tr, idx_tmp = train_test_split(
        etiquetados.index, test_size=cfg.TEST_SIZE + cfg.VAL_SIZE,
        stratify=etiquetados["etiqueta"], random_state=cfg.SEMILLA)
    prop_test = cfg.TEST_SIZE / (cfg.TEST_SIZE + cfg.VAL_SIZE)
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=prop_test,
        stratify=etiquetados.loc[idx_tmp, "etiqueta"], random_state=cfg.SEMILLA)

    df["particion"] = "sin_etiqueta"
    df.loc[idx_tr, "particion"] = "train"
    df.loc[idx_val, "particion"] = "val"
    df.loc[idx_te, "particion"] = "test"

    for nombre in ("train", "val", "test"):
        sub = df[df["particion"] == nombre]
        log.info("  %-5s: %s filas (%.1f%% positivos)", nombre, f"{len(sub):,}",
                 100 * sub["etiqueta"].mean())

    salida = df[["idx", "texto_limpio", "etiqueta", "particion"]]
    comun.guardar_parquet(salida, cfg.PQ_SPLIT_GEO, log)
    return salida


def balancear_entrenamiento(df_train: pd.DataFrame, log) -> pd.DataFrame:
    """Submuestreo SIN reemplazo de la clase mayoritaria, solo en train."""
    pos = df_train[df_train["etiqueta"] == 1]
    neg = df_train[df_train["etiqueta"] == 0]
    k = min(len(pos), len(neg))
    out = (pd.concat([pos.sample(n=k, random_state=cfg.SEMILLA, replace=False),
                      neg.sample(n=k, random_state=cfg.SEMILLA, replace=False)])
           .sample(frac=1, random_state=cfg.SEMILLA))
    log.info("Entrenamiento balanceado por submuestreo sin reemplazo: %s -> %s filas (%s por clase)",
             f"{len(df_train):,}", f"{len(out):,}", f"{k:,}")
    return out


def evaluar(y_true, y_prob, y_pred, nombre: str) -> dict:
    """Bateria de metricas que pide el asesor: precision, recall, F1,
    accuracy, mas AUC-ROC y PR-AUC (mas informativas con desbalance)."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score,
                                 confusion_matrix, balanced_accuracy_score)
    m = {
        "modelo": nombre,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            m["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            m["roc_auc"] = np.nan
            m["pr_auc"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m.update({"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)})
    return m


def umbral_optimo_f1(y_true, y_prob, log=None) -> tuple[float, float]:
    """Umbral que maximiza F1 en validacion, con dos salvaguardas.

    Se reporta junto al umbral fijo de 0.85 que usaba la version 1 sin
    justificacion documentada.

    SALVAGUARDAS. Maximizar F1 a secas puede degenerar: con pocos datos o
    con un modelo que separa mal, el optimo se va a un umbral bajisimo que
    clasifica casi todo como positivo. En una prueba real, XGBoost eligio
    0.085, con precision 0.083 y recall 0.93: inutil, y ademas contamina
    el voto del bloque ML en la etapa 03d. Por eso:

      1. la busqueda se restringe a [UMBRAL_MIN, UMBRAL_MAX];
      2. de los umbrales de ese rango se descartan los que dan una
         precision por debajo de PRECISION_MINIMA; si ninguno la alcanza,
         se avisa y se devuelve el mejor F1 del rango, para que quede
         constancia en el log de que ese modelo no es utilizable.
    """
    from sklearn.metrics import precision_recall_curve
    UMBRAL_MIN, UMBRAL_MAX, PRECISION_MINIMA = 0.20, 0.95, 0.30

    prec, rec, umb = precision_recall_curve(y_true, y_prob)
    if not len(umb):
        return 0.5, 0.0
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)

    en_rango = (umb >= UMBRAL_MIN) & (umb <= UMBRAL_MAX)
    if not en_rango.any():
        if log:
            log.warning("Ningun umbral cae en [%.2f, %.2f]; se usa 0.5.",
                        UMBRAL_MIN, UMBRAL_MAX)
        return 0.5, 0.0

    admisible = en_rango & (prec >= PRECISION_MINIMA)
    if admisible.any():
        cand = admisible
    else:
        cand = en_rango
        if log:
            log.warning("Ningun umbral alcanza precision >= %.2f en validacion. "
                        "El modelo separa mal; revisar antes de darle peso en la "
                        "regla de combinacion.", PRECISION_MINIMA)

    idx = np.where(cand)[0]
    i = idx[int(np.nanargmax(f1[idx]))]
    return float(umb[i]), float(f1[i])
