# -*- coding: utf-8 -*-
"""
ETAPA 03b — Origen colombiano, evidencia 2 (parte 1 de 2):
cuatro clasificadores clasicos sobre TF-IDF.

Que resuelve del informe del asesor
-----------------------------------
* Correccion #7 / seccion 2.3.5: "Describir algoritmos, features,
  particion train/test, validacion, metricas (precision, recall, F1,
  accuracy cuando aplique), manejo del desbalance y criterio final de
  clasificacion."

Los cinco modelos del manuscrito
--------------------------------
  1. Multinomial Naive Bayes        )
  2. Support Vector Machine lineal  )  este script (CPU)
  3. Random Forest                  )
  4. XGBoost                        )
  5. BETO (BERT en espanol) ......... s03b_modelo_transformer.py (GPU)

Representacion (features)
-------------------------
  TF-IDF sobre `texto_limpio` (lematizado, sin stopwords, sin menciones
  ni hashtags), n-gramas de 1 a 3, max_features=20,000, min_df=5,
  max_df=0.85. El vectorizador se ajusta SOLO con el conjunto de
  entrenamiento y se aplica a validacion, prueba e inferencia; ajustarlo
  sobre el corpus completo seria fuga de informacion.

Particion, desbalance y umbral: ver s03_datos_geo.py.

Salida
------
  intermedios/s03b_seed_modelos_clasicos.parquet
  resultados/tablas/s03b_metricas_modelos.csv
  resultados/tablas/s03b_matrices_confusion.csv
  resultados/figuras/s03b_matriz_confusion_<modelo>.png
  resultados/metricas/s03b_modelos_clasicos.json
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
import comun
import s03_datos_geo as datos

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ETAPA = "s03b"


def construir_modelos():
    """Los cuatro clasificadores clasicos con sus hiperparametros
    explicitos. `class_weight='balanced'` donde el estimador lo soporta
    anade una segunda capa de manejo del desbalance sobre el submuestreo."""
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "naive_bayes": {
            "estimador": MultinomialNB(alpha=1.0),
            "descripcion": "Multinomial Naive Bayes, alpha=1.0 (suavizado de Laplace)",
        },
        "svm_lineal": {
            # LinearSVC no da probabilidades; se calibra con Platt sobre
            # validacion interna para poder aplicar un umbral comparable
            # al de los demas modelos. En la version 1 este modelo no
            # tenia probabilidad y quedaba fuera de la regla de umbral.
            "estimador": CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced",
                          random_state=cfg.SEMILLA, max_iter=5000),
                method="sigmoid", cv=3),
            "descripcion": "SVM lineal C=1.0, class_weight=balanced, calibrado (Platt, cv=3)",
        },
        "random_forest": {
            "estimador": RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                random_state=cfg.SEMILLA, n_jobs=cfg.N_PROCESOS),
            "descripcion": "Random Forest, 300 arboles, min_samples_leaf=2, class_weight=balanced_subsample",
        },
        "xgboost": {
            "estimador": XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=cfg.SEMILLA, n_jobs=cfg.N_PROCESOS, tree_method="hist"),
            "descripcion": "XGBoost, 300 arboles, profundidad 6, lr=0.1, subsample=0.8",
        },
    }


def guardar_matriz_confusion(y_true, y_pred, nombre: str, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Other", "Colombia"]).plot(
        cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix — {nombre}\n(test set, natural class balance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cfg.DIR_FIGURAS / f"s03b_matriz_confusion_{nombre}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    p = comun.parser_base(__doc__)
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if comun.debe_saltar(cfg.PQ_SEED2, args.forzar, log):
        return
    if not cfg.PQ_TWITTER_PREP.exists():
        log.error("Falta %s. Ejecutar antes la etapa 02.", cfg.PQ_TWITTER_PREP.name)
        sys.exit(1)

    from sklearn.feature_extraction.text import TfidfVectorizer

    split = datos.construir_particion(log, forzar=args.forzar)
    tr = split[split["particion"] == "train"]
    va = split[split["particion"] == "val"]
    te = split[split["particion"] == "test"]
    tr_bal = datos.balancear_entrenamiento(tr, log)

    # ---------------- vectorizacion ----------------
    with comun.Cronometro(log, "ajustando TF-IDF sobre el conjunto de entrenamiento"):
        vec = TfidfVectorizer(
            ngram_range=cfg.TFIDF_NGRAMAS, max_features=cfg.TFIDF_MAX_FEATURES,
            min_df=cfg.TFIDF_MIN_DF, max_df=cfg.TFIDF_MAX_DF, sublinear_tf=True)
        X_tr = vec.fit_transform(tr_bal["texto_limpio"])
        X_va = vec.transform(va["texto_limpio"])
        X_te = vec.transform(te["texto_limpio"])
    log.info("Vocabulario TF-IDF: %s terminos", f"{len(vec.vocabulary_):,}")

    y_tr = tr_bal["etiqueta"].astype(int).values
    y_va = va["etiqueta"].astype(int).values
    y_te = te["etiqueta"].astype(int).values

    modelos = construir_modelos()
    filas_metricas, detalle, probs_corpus = [], {}, {}

    # inferencia sobre TODO el corpus, por bloques para no reventar memoria
    todos = split["texto_limpio"].values

    for nombre, spec in modelos.items():
        est = spec["estimador"]
        with comun.Cronometro(log, f"entrenando {nombre}"):
            est.fit(X_tr, y_tr)

        prob_va = est.predict_proba(X_va)[:, 1]
        prob_te = est.predict_proba(X_te)[:, 1]

        # umbral: el fijo heredado y el optimo por F1 en VALIDACION
        u_opt, f1_val = datos.umbral_optimo_f1(y_va, prob_va, log)
        u_usado = u_opt if cfg.GEO_OPTIMIZAR_UMBRAL else cfg.GEO_UMBRAL_FIJO
        log.info("%s: umbral optimo en validacion = %.3f (F1=%.3f) | umbral fijo heredado = %.2f",
                 nombre, u_opt, f1_val, cfg.GEO_UMBRAL_FIJO)

        # metricas en TEST con ambos umbrales, para que el manuscrito
        # pueda declarar cual se uso y que habria pasado con el otro
        m_opt = datos.evaluar(y_te, prob_te, (prob_te >= u_usado).astype(int),
                              nombre)
        m_opt.update({"umbral": round(u_usado, 4), "tipo_umbral": "optimo_f1_validacion"})
        m_fijo = datos.evaluar(y_te, prob_te, (prob_te >= cfg.GEO_UMBRAL_FIJO).astype(int),
                               nombre)
        m_fijo.update({"umbral": cfg.GEO_UMBRAL_FIJO, "tipo_umbral": "fijo_heredado_v1"})
        filas_metricas.extend([m_opt, m_fijo])

        guardar_matriz_confusion(y_te, (prob_te >= u_usado).astype(int), nombre, log)
        log.info("%s (test, umbral %.3f): P=%.3f R=%.3f F1=%.3f AUC=%.3f PR-AUC=%.3f",
                 nombre, u_usado, m_opt["precision"], m_opt["recall"], m_opt["f1"],
                 m_opt.get("roc_auc", np.nan), m_opt.get("pr_auc", np.nan))

        with comun.Cronometro(log, f"inferencia de {nombre} sobre el corpus completo"):
            pr = np.empty(len(todos), dtype=np.float32)
            paso = cfg.CHUNK_TWEETS   # [LOCAL] APOLO usaba 200_000
            for i in range(0, len(todos), paso):
                pr[i:i + paso] = est.predict_proba(
                    vec.transform(todos[i:i + paso]))[:, 1]
            probs_corpus[nombre] = pr

        detalle[nombre] = {
            "descripcion": spec["descripcion"],
            "umbral_optimo_validacion": u_opt,
            "f1_validacion": f1_val,
            "umbral_aplicado": u_usado,
            "metricas_test_umbral_aplicado": m_opt,
            "metricas_test_umbral_fijo": m_fijo,
        }

    # ---------------- salida ----------------
    met = pd.DataFrame(filas_metricas)[
        ["modelo", "tipo_umbral", "umbral", "accuracy", "balanced_accuracy",
         "precision", "recall", "f1", "roc_auc", "pr_auc", "vp", "fp", "fn", "vn"]
    ].round(4)
    comun.guardar_tabla(met, "s03b_metricas_modelos", log)
    log.info("\n%s", met.to_string(index=False))

    salida = pd.DataFrame({"idx": split["idx"].values})
    for nombre, pr in probs_corpus.items():
        u = detalle[nombre]["umbral_aplicado"]
        salida[f"prob_{nombre}"] = pr
        salida[f"pred_{nombre}"] = (pr >= u).astype(np.int8)

    cols_pred = [c for c in salida.columns if c.startswith("pred_")]
    salida["votos_clasicos"] = salida[cols_pred].sum(axis=1).astype(np.int8)
    salida["n_modelos_clasicos"] = len(cols_pred)
    comun.guardar_parquet(salida, cfg.PQ_SEED2, log)

    concordancia = (salida[cols_pred].sum(axis=1)
                    .value_counts().sort_index()
                    .rename_axis("modelos_que_dicen_colombia")
                    .reset_index(name="n_tuits"))
    comun.guardar_tabla(concordancia, "s03b_concordancia_entre_modelos", log)
    log.info("Concordancia entre los 4 modelos clasicos:\n%s",
             concordancia.to_string(index=False))

    comun.guardar_metricas({
        "features": {
            "representacion": "TF-IDF sublinear sobre texto_limpio",
            "ngramas": list(cfg.TFIDF_NGRAMAS),
            "max_features": cfg.TFIDF_MAX_FEATURES,
            "min_df": cfg.TFIDF_MIN_DF, "max_df": cfg.TFIDF_MAX_DF,
            "vocabulario_efectivo": len(vec.vocabulary_),
            "ajustado_solo_con": "conjunto de entrenamiento balanceado",
        },
        "particion": {
            "train": int(len(tr)), "train_balanceado": int(len(tr_bal)),
            "val": int(len(va)), "test": int(len(te)),
            "sin_etiqueta": int((split["particion"] == "sin_etiqueta").sum()),
            "estratificada": True, "semilla": cfg.SEMILLA,
        },
        "desbalance": {
            "razon_original_neg_pos": float((y_te == 0).sum() / max((y_te == 1).sum(), 1)),
            "estrategia": "submuestreo sin reemplazo solo en train + class_weight donde aplica",
            "metricas_reportadas_sobre": "test con distribucion natural",
        },
        "modelos": detalle,
    }, "s03b_modelos_clasicos", log)

    log.info("ETAPA 03b COMPLETADA (4 modelos clasicos).")


if __name__ == "__main__":
    main()
