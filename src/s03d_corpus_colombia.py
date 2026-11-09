# -*- coding: utf-8 -*-
"""
ETAPA 03d — Regla de combinacion y corpus colombiano definitivo.

Que resuelve del informe del asesor
-----------------------------------
* Seccion 2.3.5, ultima frase: "...y forma de combinar el gazetteer,
  lexicon y grafo con el clasificador automatico."

  Esta etapa es esa regla, escrita de forma explicita y auditable.

La regla
--------
Cada tuit recibe hasta TRES votos independientes:

  V1  gazetteer + lexicon + Location compatible   (etapa 03a)
  V2  bloque de modelos ML: cuenta como UN voto si al menos
      config.GEO_ML_MIN_MODELOS de los 5 modelos lo clasifican como
      colombiano                                   (etapa 03b, 2 partes)
  V3  propagacion sobre el grafo de respuestas     (etapa 03c)

  Criterio final (config.GEO_REGLA_COMBINACION = "mayoria"):
      colombiano  <=>  V1 + V2 + V3  >=  config.GEO_MIN_VOTOS   (=2)

Por que mayoria y no union
--------------------------
La union maximiza cobertura pero acumula los falsos positivos de las tres
evidencias, y el gazetteer los tiene de sobra por toponimos ambiguos. La
mayoria exige que dos senales independientes coincidan, lo que sube la
precision a costa de recall. Dado que el paper describe el tono del
discurso y no estima prevalencias poblacionales, un corpus mas pequeno y
mas limpio es preferible a uno grande y contaminado.

El script calcula ademas el corpus que habrian producido las reglas
alternativas ("union", "gazetteer_o_ml") y guarda la comparacion, para
que el manuscrito pueda presentar un analisis de sensibilidad y para que
la eleccion no parezca arbitraria.

Salida
------
  intermedios/s03d_corpus_colombia.parquet   <- corpus filtrado, no solo banderas
  resultados/tablas/s03d_regla_combinacion.csv
  resultados/tablas/s03d_sensibilidad_reglas.csv
  resultados/metricas/s03d_corpus_colombia.json
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
import comun

ETAPA = "s03d"


def main() -> None:
    p = comun.parser_base(__doc__)
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if comun.debe_saltar(cfg.PQ_COLOMBIA, args.forzar, log):
        return
    for req in (cfg.PQ_TWITTER_PREP, cfg.PQ_SEED1, cfg.PQ_SEED2, cfg.PQ_SEED3):
        if not req.exists():
            log.error("Falta %s. Ejecutar las etapas 02, 03a, 03b y 03c.", req.name)
            sys.exit(1)

    # [LOCAL] No se carga el corpus completo: primero se calculan las
    # banderas (que son solo booleanos, unos pocos MB) y despues se
    # recorre el parquet por bloques escribiendo unicamente las filas
    # seleccionadas. La version de APOLO cargaba las 1.44M de filas con
    # todas sus columnas, varios GB.
    import pyarrow as pa
    import pyarrow.parquet as pq

    s1 = pd.read_parquet(cfg.PQ_SEED1)
    n = len(s1)
    s2 = pd.read_parquet(cfg.PQ_SEED2)
    s3 = pd.read_parquet(cfg.PQ_SEED3)

    # ---------------- voto 2: bloque de modelos ML ----------------
    cols_pred = [c for c in s2.columns if c.startswith("pred_")]
    n_modelos = len(cols_pred)
    votos_ml = s2[cols_pred].sum(axis=1).values
    if cfg.PQ_SEED2B.exists():
        b = pd.read_parquet(cfg.PQ_SEED2B)
        votos_ml = votos_ml + b["pred_beto"].values
        n_modelos += 1
        log.info("BETO incluido: el bloque ML son %d modelos.", n_modelos)
    else:
        log.warning("No existe %s: el bloque ML son solo %d modelos (faltan los "
                    "cinco que declara el manuscrito). Ejecutar "
                    "s03b_modelo_transformer.py antes del envio.",
                    cfg.PQ_SEED2B.name, n_modelos)

    v1 = s1["seed_1"].values.astype(bool)
    v2 = (votos_ml >= cfg.GEO_ML_MIN_MODELOS)
    v3 = s3["seed_3"].values.astype(bool)
    total_votos = v1.astype(int) + v2.astype(int) + v3.astype(int)

    # ---------------- reglas ----------------
    reglas = {
        "mayoria": total_votos >= cfg.GEO_MIN_VOTOS,
        "union": total_votos >= 1,
        "gazetteer_o_ml": v1 | v2,
        "unanimidad": total_votos == 3,
        "solo_gazetteer": v1,
        "solo_ml": v2,
        "solo_grafo": v3,
    }
    sens = pd.DataFrame([
        {"regla": k, "n_tuits": int(m.sum()), "pct_del_corpus": round(100 * m.mean(), 2)}
        for k, m in reglas.items()
    ]).sort_values("n_tuits", ascending=False)
    comun.guardar_tabla(sens, "s03d_sensibilidad_reglas", log)
    log.info("Sensibilidad a la regla de combinacion:\n%s", sens.to_string(index=False))

    es_colombiano = reglas[cfg.GEO_REGLA_COMBINACION]
    log.info("REGLA APLICADA: '%s' (>=%d votos de 3) -> %s tuits colombianos (%.2f%%)",
             cfg.GEO_REGLA_COMBINACION, cfg.GEO_MIN_VOTOS,
             f"{int(es_colombiano.sum()):,}", 100 * es_colombiano.mean())

    # ---------------- desglose de votos y solapamiento ----------------
    desglose = pd.DataFrame([
        {"evidencia": "V1 gazetteer+lexicon+location", "n": int(v1.sum())},
        {"evidencia": "V2 bloque ML (>=%d de %d modelos)" % (cfg.GEO_ML_MIN_MODELOS, n_modelos),
         "n": int(v2.sum())},
        {"evidencia": "V3 grafo de respuestas", "n": int(v3.sum())},
        {"evidencia": "exactamente 1 voto", "n": int((total_votos == 1).sum())},
        {"evidencia": "exactamente 2 votos", "n": int((total_votos == 2).sum())},
        {"evidencia": "los 3 votos", "n": int((total_votos == 3).sum())},
        {"evidencia": "ningun voto", "n": int((total_votos == 0).sum())},
        {"evidencia": "CORPUS COLOMBIANO FINAL", "n": int(es_colombiano.sum())},
    ])
    desglose["pct_del_corpus"] = (desglose["n"] / n * 100).round(2)
    comun.guardar_tabla(desglose, "s03d_regla_combinacion", log)
    log.info("\n%s", desglose.to_string(index=False))

    # acuerdo por pares entre evidencias (Jaccard)
    ev = {"gazetteer": v1, "modelos_ml": v2, "grafo": v3}
    pares = []
    for a, b in combinations(ev, 2):
        inter = int((ev[a] & ev[b]).sum())
        union = int((ev[a] | ev[b]).sum())
        pares.append({"evidencia_a": a, "evidencia_b": b, "interseccion": inter,
                      "union": union, "jaccard": round(inter / union, 4) if union else np.nan})
    comun.guardar_tabla(pd.DataFrame(pares), "s03d_acuerdo_entre_evidencias", log)

    # ---------------- corpus final ----------------
    fuente = pq.ParquetFile(cfg.PQ_TWITTER_PREP)
    if fuente.metadata.num_rows != n:
        log.error("El parquet preprocesado tiene %s filas y las banderas %s. "
                  "Rehacer las etapas 02 y 03 con --forzar.",
                  f"{fuente.metadata.num_rows:,}", f"{n:,}")
        sys.exit(1)

    ruta_tmp = cfg.PQ_COLOMBIA.with_suffix(".tmp.parquet")
    escritor = esquema = None
    n_col = 0
    desde = 0
    with comun.Cronometro(log, "escribiendo el corpus colombiano por bloques"):
        try:
            for lote in fuente.iter_batches(batch_size=cfg.CHUNK_TWEETS):
                hasta = desde + lote.num_rows
                sel = es_colombiano[desde:hasta]
                if sel.any():
                    d = lote.to_pandas()[sel].copy()
                    d["voto_gazetteer"] = v1[desde:hasta][sel]
                    d["voto_modelos_ml"] = v2[desde:hasta][sel]
                    d["voto_grafo"] = v3[desde:hasta][sel]
                    d["n_votos_colombia"] = total_votos[desde:hasta][sel]
                    t = pa.Table.from_pandas(d.reset_index(drop=True),
                                             preserve_index=False)
                    if escritor is None:
                        esquema = t.schema
                        escritor = pq.ParquetWriter(ruta_tmp, esquema,
                                                    compression="snappy")
                    else:
                        t = t.cast(esquema)
                    escritor.write_table(t, row_group_size=cfg.PARQUET_ROW_GROUP)
                    n_col += len(d)
                    del d, t
                desde = hasta
        finally:
            if escritor is not None:
                escritor.close()

    if escritor is None:
        log.error("La regla '%s' no selecciono ningun tuit. Revisar las etapas previas.",
                  cfg.GEO_REGLA_COMBINACION)
        sys.exit(1)
    import os
    os.replace(ruta_tmp, cfg.PQ_COLOMBIA)
    log.info("Guardado %s (%s filas)", cfg.PQ_COLOMBIA.name, f"{n_col:,}")
    col = pd.read_parquet(cfg.PQ_COLOMBIA, columns=["autor", "fecha"])
    comun.registrar_funnel(
        ETAPA, "E10_corpus_colombia", n_col,
        f"Origen colombiano por regla '{cfg.GEO_REGLA_COMBINACION}' "
        f"(>= {cfg.GEO_MIN_VOTOS} de 3 evidencias independientes)")

    autores = col["autor"].nunique()
    log.info("Corpus colombiano: %s tuits de %s autores distintos, %s a %s",
             f"{n_col:,}", f"{autores:,}",
             col["fecha"].min().date(), col["fecha"].max().date())

    comun.guardar_metricas({
        "regla_aplicada": cfg.GEO_REGLA_COMBINACION,
        "min_votos": cfg.GEO_MIN_VOTOS,
        "n_modelos_en_bloque_ml": n_modelos,
        "min_modelos_para_voto_ml": cfg.GEO_ML_MIN_MODELOS,
        "beto_incluido": cfg.PQ_SEED2B.exists(),
        "n_corpus_entrada": n,
        "n_corpus_colombia": int(n_col),
        "pct_retenido": round(100 * n_col / n, 2),
        "autores_distintos": int(autores),
        "desglose_votos": desglose.to_dict("records"),
        "sensibilidad_reglas": sens.to_dict("records"),
        "acuerdo_entre_evidencias": pares,
        "advertencia": ("Ninguna de las tres evidencias tiene validacion humana del "
                        "origen geografico. Las metricas de la etapa 03b se apoyan en "
                        "supervision distante a partir del campo Location. Declararlo "
                        "en Limitations."),
    }, "s03d_corpus_colombia", log)

    log.info("ETAPA 03d COMPLETADA.")


if __name__ == "__main__":
    main()
