# -*- coding: utf-8 -*-
"""
ETAPA 03c — Origen colombiano, evidencia 3: propagacion sobre el grafo de
respuestas.

Que resuelve del informe del asesor
-----------------------------------
* Seccion 2.3.5: "...y forma de combinar el gazetteer, lexicon y GRAFO con
  el clasificador automatico."

Intuicion
---------
La conversacion en Twitter es homofila por pais: quien responde
repetidamente a cuentas colombianas suele ser colombiano, aunque su tuit
no nombre ningun toponimo ni declare ubicacion. Esta evidencia recupera
justamente a esos usuarios, que son invisibles para el gazetteer y
dificiles para el clasificador de texto.

Que se construye
----------------
  Nodos  = autores.
  Aristas= "A responde a B", con peso = numero de respuestas.
           Solo cuentan los vinculos con al menos
           config.GRAFO_MIN_INTERACCIONES respuestas, para que un unico
           reply casual no arrastre a un autor entero.

  Semilla inicial = autores con evidencia previa (gazetteer o modelos ML).
  Un autor entra a la semilla si al menos la mitad de sus tuits con
  evidencia lo son; asi una sola coincidencia espuria en un autor muy
  prolifico no lo convierte en semilla.

Propagacion (label propagation acotada)
---------------------------------------
  Se repite config.GRAFO_ITERACIONES veces:
      un autor no marcado pasa a marcado si la proporcion de sus vecinos
      ya marcados (ponderada por numero de interacciones) supera
      config.GRAFO_UMBRAL_VECINOS.

  Se acota a pocas iteraciones a proposito: la propagacion sin limite
  termina marcando el grafo entero, que es el modo tipico de fallo de
  esta familia de metodos. Se reporta cuantos autores y tuits se anaden
  en cada ronda para que el efecto sea auditable.

Limitacion que debe ir al manuscrito: este metodo NO tiene validacion
independiente. Se apoya en la evidencia previa, asi que hereda sus
errores. Por eso vale como un voto mas y nunca como criterio unico.

Salida
------
  intermedios/s03c_seed_grafo.parquet
  resultados/tablas/s03c_propagacion_grafo.csv
  resultados/metricas/s03c_grafo.json
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
import comun

ETAPA = "s03c"


def normaliza_handle(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def main() -> None:
    p = comun.parser_base(__doc__)
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if comun.debe_saltar(cfg.PQ_SEED3, args.forzar, log):
        return
    for req in (cfg.PQ_TWITTER_PREP, cfg.PQ_SEED1):
        if not req.exists():
            log.error("Falta %s. Ejecutar las etapas previas.", req.name)
            sys.exit(1)

    base = pd.read_parquet(cfg.PQ_TWITTER_PREP, columns=["autor", "responde_a"])
    seed1 = pd.read_parquet(cfg.PQ_SEED1, columns=["seed_1"])
    base["autor_n"] = normaliza_handle(base["autor"])
    base["responde_n"] = normaliza_handle(base["responde_a"])
    base["evidencia"] = seed1["seed_1"].values

    # incorporar el voto de los modelos ML si ya existen
    fuentes_ml = []
    if cfg.PQ_SEED2.exists():
        ml = pd.read_parquet(cfg.PQ_SEED2)
        cols = [c for c in ml.columns if c.startswith("pred_")]
        base["evidencia"] |= (ml[cols].sum(axis=1) >= cfg.GEO_ML_MIN_MODELOS).values
        fuentes_ml.append(f"modelos clasicos ({len(cols)})")
    if cfg.PQ_SEED2B.exists():
        bt = pd.read_parquet(cfg.PQ_SEED2B)
        base["evidencia"] |= bt["pred_beto"].astype(bool).values
        fuentes_ml.append("BETO")
    log.info("Evidencia previa combinada de: gazetteer%s",
             (" + " + " + ".join(fuentes_ml)) if fuentes_ml else " (solo)")
    log.info("Tuits con evidencia previa: %s (%.2f%%)",
             f"{int(base['evidencia'].sum()):,}", 100 * base["evidencia"].mean())

    # ---------------- semilla de autores ----------------
    por_autor = base.groupby("autor_n")["evidencia"].agg(["sum", "size"])
    por_autor = por_autor[por_autor.index != ""]
    proporcion = por_autor["sum"] / por_autor["size"]
    semilla = set(proporcion[proporcion >= 0.5].index)
    log.info("Autores distintos: %s | autores semilla (>=50%% de sus tuits con evidencia): %s",
             f"{len(por_autor):,}", f"{len(semilla):,}")

    # ---------------- grafo de respuestas ----------------
    aristas = base[(base["responde_n"] != "") & (base["autor_n"] != "")]
    pares = aristas.groupby(["autor_n", "responde_n"]).size()
    pares = pares[pares >= cfg.GRAFO_MIN_INTERACCIONES]
    log.info("Aristas con >=%d respuestas: %s", cfg.GRAFO_MIN_INTERACCIONES, f"{len(pares):,}")

    vecinos = defaultdict(list)   # nodo -> [(vecino, peso)]
    for (a, b), w in pares.items():
        vecinos[a].append((b, int(w)))
        vecinos[b].append((a, int(w)))
    log.info("Nodos con al menos una arista: %s", f"{len(vecinos):,}")

    # ---------------- propagacion ----------------
    marcados = set(semilla)
    historial = [{"iteracion": 0, "autores_marcados": len(marcados),
                  "autores_nuevos": len(marcados),
                  "descripcion": "semilla inicial (gazetteer + modelos ML)"}]

    for it in range(1, cfg.GRAFO_ITERACIONES + 1):
        nuevos = set()
        for nodo, vs in vecinos.items():
            if nodo in marcados:
                continue
            total = sum(w for _, w in vs)
            si = sum(w for v, w in vs if v in marcados)
            if total and si / total >= cfg.GRAFO_UMBRAL_VECINOS:
                nuevos.add(nodo)
        marcados |= nuevos
        historial.append({"iteracion": it, "autores_marcados": len(marcados),
                          "autores_nuevos": len(nuevos),
                          "descripcion": f"propagacion, umbral {cfg.GRAFO_UMBRAL_VECINOS}"})
        log.info("Iteracion %d: +%s autores (total %s)", it, f"{len(nuevos):,}", f"{len(marcados):,}")
        if not nuevos:
            log.info("Sin autores nuevos: la propagacion converge, se detiene.")
            break

    hist = pd.DataFrame(historial)
    hist["tuits_cubiertos"] = [
        int(base["autor_n"].isin(
            set(list(semilla)) if r == 0 else marcados).sum())
        for r in hist["iteracion"]
    ]
    comun.guardar_tabla(hist, "s03c_propagacion_grafo", log)

    seed3 = base["autor_n"].isin(marcados).values
    solo_grafo = int((seed3 & ~base["evidencia"].values).sum())
    log.info("Seed_3 = %s tuits; de ellos %s NO tenian evidencia previa "
             "(es la aportacion neta del grafo)",
             f"{int(seed3.sum()):,}", f"{solo_grafo:,}")

    comun.guardar_parquet(pd.DataFrame({"seed_3": seed3}), cfg.PQ_SEED3, log)
    comun.registrar_funnel(ETAPA, "E09_seed_grafo", int(seed3.sum()),
                           "Evidencia 3: propagacion sobre el grafo de respuestas")

    comun.guardar_metricas({
        "autores_distintos": int(len(por_autor)),
        "autores_semilla": len(semilla),
        "autores_marcados_final": len(marcados),
        "aristas": int(len(pares)),
        "nodos_en_grafo": len(vecinos),
        "parametros": {
            "iteraciones": cfg.GRAFO_ITERACIONES,
            "min_interacciones": cfg.GRAFO_MIN_INTERACCIONES,
            "umbral_vecinos": cfg.GRAFO_UMBRAL_VECINOS,
            "criterio_semilla": ">=50% de los tuits del autor con evidencia previa",
        },
        "historial": historial,
        "n_seed_3": int(seed3.sum()),
        "aporte_neto_del_grafo": solo_grafo,
        "fuentes_evidencia_previa": ["gazetteer"] + fuentes_ml,
    }, "s03c_grafo", log)

    log.info("ETAPA 03c COMPLETADA.")


if __name__ == "__main__":
    main()
