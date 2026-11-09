# -*- coding: utf-8 -*-
"""
ETAPA 06 — Componente de encuesta: iCARE Colombia.

Que resuelve del informe del asesor
-----------------------------------
* Correccion #3 / seccion 2.3.2: "Especificar con exactitud: periodo
  usado, numero total recogido, numero incluido en la ventana comun,
  criterios de inclusion/exclusion, waves involucradas, tratamiento de
  missing data y denominadores efectivos por variable."
* Correcciones #12 y #13 / seccion 2.4: "Reconstruir la Tabla 1 desde la
  base final. Reportar n valido por variable y explicar por que algunos
  items tienen denominadores diferentes por ondas o datos faltantes."

HALLAZGO QUE OBLIGA A REESCRIBIR METHODS
----------------------------------------
Verificado sobre df_icare_original.xlsx:

  N total del dataset Colombia (w1-w6) .................. 2,301
  N con startdate entre 2020-03-31 y 2020-08-18 ......... 2,129
  N que respondio al menos un item trustinf .............   125
  N con trustinf Y dentro de la ventana de Twitter ......     0

  Los items trustinf_* solo existen en las OLAS 5 y 6, recolectadas entre
  2020-09-16 y 2020-12-06. El corpus de Twitter cubre 2020-03-31 a
  2020-08-18. No hay ni un solo dia de solapamiento.

  El "2,130" que aparecia en el manuscrito es el N de la ventana, pero
  NINGUNA de esas 2,129 personas contesto los items de confianza. De ahi
  que los denominadores de la Tabla 1 fueran incoherentes: el
  denominador real de las variables de confianza es ~125.

  DECISION EDITORIAL TOMADA: el paper declara DOS periodos distintos y
  elimina toda afirmacion de "ventana comun". Componente de encuesta:
  septiembre-diciembre 2020. Componente digital: marzo-agosto 2020.
  Por eso config.ICARE_FILTRAR_POR_VENTANA = False.

  El script emite igualmente el N de la ventana para que la trazabilidad
  del "2,130" quede documentada.

Por que el denominador de Employment era 164
--------------------------------------------
`emplstat_sq001..sq008` NO es una variable categorica unica: es una
pregunta de SELECCION MULTIPLE con ocho casillas 0/1 independientes. En
el manuscrito anterior se sumaron los "1" de cada casilla como si fueran
categorias excluyentes, y esa suma dio 164. Aqui se reporta como
multi-respuesta: cada opcion con su propio numerador y un denominador
comun (quienes contestaron el bloque), y se advierte que los porcentajes
no suman 100.

Binarizacion de la confianza
----------------------------
  Escala: 1 confio mucho | 2 confio algo | 3 no confio | 4 no confio nada
          5 no se / prefiero no responder
  Regla:  {1,2} -> 1 (confia) ; {3,4} -> 0 (no confia) ; {5} -> MISSING

  El 5 es no-respuesta, no desconfianza. Contarlo como 0 inflaria el
  distrust. Se reporta cuantos 5 hubo por item para que la decision sea
  auditable y reproducible.

Precision de las estimaciones
-----------------------------
Con n~125 toda proporcion se reporta con intervalo de Wilson al 95%. Los
intervalos son anchos (del orden de +-9 puntos) y el ranking de
categorias por desconfianza debe presentarse como ordenamiento
indicativo, no como estimacion puntual.

Salida
------
  intermedios/s06_icare.parquet
  resultados/tablas/s06_flujo_icare.csv
  resultados/tablas/s06_tabla1_sociodemograficas.csv
  resultados/tablas/s06_tabla1_empleo_multirespuesta.csv
  resultados/tablas/s06_confianza_por_fuente.csv
  resultados/tablas/s06_fuentes_de_informacion_infoso.csv
  resultados/metricas/s06_icare.json
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

ETAPA = "s06"


def binarizar_confianza(s: pd.Series) -> pd.Series:
    """{1,2}->1 ; {3,4}->0 ; {5}->NaN ; cualquier otro valor->NaN."""
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s.isin(cfg.TRUST_VALORES_SI)] = 1.0
    out[s.isin(cfg.TRUST_VALORES_NO)] = 0.0
    return out


def grupo_edad(v) -> str:
    if pd.isna(v):
        return np.nan
    for lo, hi in cfg.ICARE_GRUPOS_EDAD:
        if lo <= v <= hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    return "<18"


def main() -> None:
    p = comun.parser_base(__doc__)
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if comun.debe_saltar(cfg.PQ_ICARE, args.forzar, log):
        return
    if not cfg.ARCHIVO_ICARE.exists():
        log.error("Falta %s", cfg.ARCHIVO_ICARE)
        sys.exit(1)

    d = pd.read_excel(cfg.ARCHIVO_ICARE)
    n_total = len(d)
    log.info("Dataset iCARE Colombia: %s filas, %s columnas", f"{n_total:,}", d.shape[1])

    d["_fecha"] = pd.to_datetime(d[cfg.ICARE_COL_FECHA], errors="coerce")
    cols_trust = [v["trustinf"] for v in cfg.CATEGORIAS.values() if v["trustinf"]]
    cols_trust = [c for c in cols_trust if c in d.columns]
    d["_respondio_trust"] = d[cols_trust].notna().any(axis=1)

    ini, fin = (pd.Timestamp(cfg.ICARE_VENTANA_REFERENCIA[0]),
                pd.Timestamp(cfg.ICARE_VENTANA_REFERENCIA[1]) + pd.Timedelta(days=1))
    en_ventana = d["_fecha"].between(ini, fin, inclusive="left")

    # ---------------- flujo ----------------
    flujo = pd.DataFrame([
        {"paso": "Total recolectado en Colombia (olas 1-6)", "n": n_total},
        {"paso": f"Con fecha de inicio entre {cfg.ICARE_VENTANA_REFERENCIA[0]} y "
                 f"{cfg.ICARE_VENTANA_REFERENCIA[1]} (ventana del corpus de Twitter)",
         "n": int(en_ventana.sum())},
        {"paso": "Respondio al menos un item de confianza (trustinf)",
         "n": int(d["_respondio_trust"].sum())},
        {"paso": "Respondio trustinf Y esta dentro de la ventana de Twitter",
         "n": int((en_ventana & d["_respondio_trust"]).sum())},
    ])
    comun.guardar_tabla(flujo, "s06_flujo_icare", log)
    log.info("\n%s", flujo.to_string(index=False))

    if int((en_ventana & d["_respondio_trust"]).sum()) == 0:
        log.warning("")
        log.warning("!!! NO HAY SOLAPAMIENTO TEMPORAL entre los items de confianza y el")
        log.warning("!!! corpus de Twitter. El manuscrito NO puede afirmar ventana comun.")
        log.warning("!!! Methods debe declarar los dos periodos por separado.")
        log.warning("")

    olas = sorted(d.loc[d["_respondio_trust"], cfg.ICARE_COL_OLA].dropna().unique().tolist())
    f_min = d.loc[d["_respondio_trust"], "_fecha"].min()
    f_max = d.loc[d["_respondio_trust"], "_fecha"].max()
    log.info("Olas con items de confianza: %s | recolectadas del %s al %s",
             olas, f_min.date() if pd.notna(f_min) else "?",
             f_max.date() if pd.notna(f_max) else "?")

    # ---------------- muestra analitica ----------------
    if cfg.ICARE_FILTRAR_POR_VENTANA:
        muestra = d[en_ventana & d["_respondio_trust"]].copy()
        criterio = "trustinf respondido Y dentro de la ventana de Twitter"
    else:
        muestra = d[d["_respondio_trust"]].copy()
        criterio = ("trustinf respondido (sin filtro de ventana: no hay solapamiento "
                    "temporal con el corpus de Twitter)")
    n_m = len(muestra)
    log.info("Muestra analitica de confianza: %s participantes (%s)", f"{n_m:,}", criterio)
    if n_m == 0:
        log.error("Muestra analitica vacia. Revisar los criterios.")
        sys.exit(1)

    comun.registrar_funnel(ETAPA, "I01_icare_total", n_total, "Total recolectado en Colombia")
    comun.registrar_funnel(ETAPA, "I02_icare_trust", n_m,
                           "Participantes con items de confianza respondidos")

    # ---------------- Tabla 1 ----------------
    filas_t1 = []
    for var, spec in cfg.ICARE_SOCIODEMOGRAFICAS.items():
        if var not in muestra.columns:
            continue
        s = muestra[var]
        valid = int(s.notna().sum())
        for cod, etiqueta in spec["niveles"].items():
            k = int((s == cod).sum())
            if k == 0 and cod not in s.dropna().unique():
                continue
            f = comun.fila_proporcion(etiqueta, k, valid)
            f.update({"variable": spec["etiqueta_en"], "faltantes": n_m - valid})
            filas_t1.append(f)
        otros = int(s.notna().sum() - sum(int((s == c).sum()) for c in spec["niveles"]))
        if otros:
            f = comun.fila_proporcion("Other/unmapped code", otros, valid)
            f.update({"variable": spec["etiqueta_en"], "faltantes": n_m - valid})
            filas_t1.append(f)

    if cfg.ICARE_COL_EDAD in muestra.columns:
        edad = pd.to_numeric(muestra[cfg.ICARE_COL_EDAD], errors="coerce")
        valid = int(edad.notna().sum())
        g = edad.map(grupo_edad)
        for etiqueta in sorted(g.dropna().unique()):
            f = comun.fila_proporcion(etiqueta, int((g == etiqueta).sum()), valid)
            f.update({"variable": "Age group (years)", "faltantes": n_m - valid})
            filas_t1.append(f)
        log.info("Edad: media %.1f, DE %.1f, rango %.0f-%.0f, n valido %d",
                 edad.mean(), edad.std(), edad.min(), edad.max(), valid)

    t1 = pd.DataFrame(filas_t1)[
        ["variable", "categoria", "n", "valid_n", "faltantes", "pct", "ic95_inf", "ic95_sup"]]
    comun.guardar_tabla(t1, "s06_tabla1_sociodemograficas", log)
    log.info("\n%s", t1.to_string(index=False))
    log.info("NOTA para el manuscrito: la columna valid_n varia entre variables porque "
             "los items no se preguntaron en todas las olas y hay no-respuesta parcial. "
             "Es exactamente lo que pide la correccion #12.")

    # ---------------- empleo (multi-respuesta) ----------------
    cols_emp = [c for c in muestra.columns if c.startswith(cfg.ICARE_EMPLEO_PREFIJO)]
    if cols_emp:
        contesto = muestra[cols_emp].notna().any(axis=1)
        den = int(contesto.sum())
        filas_emp = []
        for c in cols_emp:
            k = int((muestra[c] == 1).sum())
            f = comun.fila_proporcion(cfg.ICARE_EMPLEO_NIVELES.get(c, c), k, den)
            f["variable"] = cfg.ICARE_EMPLEO_ETIQUETA
            f["item"] = c
            filas_emp.append(f)
        emp = pd.DataFrame(filas_emp)[
            ["variable", "item", "categoria", "n", "valid_n", "pct", "ic95_inf", "ic95_sup"]]
        emp.attrs["nota"] = "pregunta de seleccion multiple; los porcentajes no suman 100"
        comun.guardar_tabla(emp, "s06_tabla1_empleo_multirespuesta", log)
        suma = int(emp["n"].sum())
        log.info("\n%s", emp.to_string(index=False))
        log.info("Empleo: denominador comun = %d participantes que contestaron el bloque; "
                 "la suma de casillas marcadas es %d. Esa suma es la que en el manuscrito "
                 "anterior se reporto por error como denominador.", den, suma)

    # ---------------- confianza por fuente ----------------
    filas_tr, resumen_missing = [], []
    for clave, spec in cfg.CATEGORIAS.items():
        col = spec["trustinf"]
        if not col or col not in muestra.columns:
            continue
        cruda = muestra[col]
        b = binarizar_confianza(cruda)
        valid = int(b.notna().sum())
        confia = int((b == 1).sum())
        nocon = int((b == 0).sum())
        n_ns = int(cruda.isin(cfg.TRUST_VALORES_MISSING).sum())
        n_na = int(cruda.isna().sum())

        f_t = comun.fila_proporcion(spec["etiqueta_en"], confia, valid)
        f_t.update({"clave": clave, "variable": col, "medida": "Trust",
                    "no_se_excluidos": n_ns, "sin_respuesta": n_na,
                    "observable_en_twitter": spec["observable_twitter"]})
        f_d = comun.fila_proporcion(spec["etiqueta_en"], nocon, valid)
        f_d.update({"clave": clave, "variable": col, "medida": "Distrust",
                    "no_se_excluidos": n_ns, "sin_respuesta": n_na,
                    "observable_en_twitter": spec["observable_twitter"]})
        filas_tr.extend([f_t, f_d])
        resumen_missing.append({"variable": col, "categoria": spec["etiqueta_en"],
                                "valid_n": valid, "no_se_5": n_ns, "sin_respuesta": n_na})

        muestra[f"trust_bin_{clave}"] = b

    tr = pd.DataFrame(filas_tr)[
        ["clave", "variable", "categoria", "medida", "n", "valid_n", "pct",
         "ic95_inf", "ic95_sup", "no_se_excluidos", "sin_respuesta", "observable_en_twitter"]]
    comun.guardar_tabla(tr, "s06_confianza_por_fuente", log)

    dis = (tr[tr["medida"] == "Distrust"].sort_values("pct", ascending=False)
           .reset_index(drop=True))
    log.info("\nDESCONFIANZA DECLARADA, ordenada de mayor a menor:\n%s",
             dis[["categoria", "n", "valid_n", "pct", "ic95_inf", "ic95_sup"]]
             .to_string(index=False))
    log.info("Los IC95%% son anchos por el tamano de muestra (n~%d). El ranking debe "
             "presentarse como ordenamiento indicativo, no como estimacion puntual.", n_m)

    comun.guardar_tabla(pd.DataFrame(resumen_missing), "s06_missing_por_item_confianza", log)

    # ---------------- infoso: fuentes usadas ----------------
    # Variable complementaria. A diferencia de trustinf, SI esta dentro de
    # la ventana del corpus de Twitter. Mide uso/importancia, no confianza:
    # no es un sustituto y no debe presentarse como tal.
    cols_info = [c for c in d.columns if c.startswith("infoso_")]
    if cols_info:
        sub = d[en_ventana]
        opciones = {v["infoso"]: v["etiqueta_en"] for v in cfg.CATEGORIAS.values()
                    if v.get("infoso")}
        opciones.setdefault(2, "Workplace")
        opciones.setdefault(10, "Other people or groups via social media or the internet")
        primero = pd.to_numeric(sub["infoso_1"], errors="coerce")
        den = int(primero.notna().sum())
        filas_inf = []
        for cod, etiqueta in sorted(opciones.items()):
            filas_inf.append(comun.fila_proporcion(etiqueta, int((primero == cod).sum()), den))
        inf = pd.DataFrame(filas_inf).sort_values("pct", ascending=False)
        inf["nota"] = "fuente clasificada en primer lugar (infoso_1), dentro de la ventana de Twitter"
        comun.guardar_tabla(inf, "s06_fuentes_de_informacion_infoso", log)
        log.info("\nFuente de informacion mas importante (infoso_1, n=%d, DENTRO de la "
                 "ventana de Twitter):\n%s", den,
                 inf[["categoria", "n", "valid_n", "pct"]].to_string(index=False))
        log.info("infoso mide USO, no confianza. Es complementaria, no sustituta de trustinf.")

    # ---------------- salida ----------------
    muestra["_grupo_edad"] = pd.to_numeric(
        muestra.get(cfg.ICARE_COL_EDAD), errors="coerce").map(grupo_edad)
    guardar_cols = ([c for c in ("id", "wave", "sex", "age", "edu") if c in muestra.columns]
                    + ["_fecha", "_grupo_edad"] + cols_trust
                    + [c for c in muestra.columns if c.startswith("trust_bin_")]
                    + cols_emp)
    comun.guardar_parquet(muestra[guardar_cols].reset_index(drop=True), cfg.PQ_ICARE, log)

    comun.guardar_metricas({
        "n_total_dataset": n_total,
        "n_en_ventana_twitter": int(en_ventana.sum()),
        "n_respondio_trustinf": int(d["_respondio_trust"].sum()),
        "n_trustinf_y_en_ventana": int((en_ventana & d["_respondio_trust"]).sum()),
        "olas_con_trustinf": olas,
        "periodo_recoleccion_trustinf": [str(f_min), str(f_max)],
        "periodo_corpus_twitter": [cfg.TWITTER_FECHA_INICIO, cfg.TWITTER_FECHA_FIN],
        "hay_solapamiento_temporal": bool((en_ventana & d["_respondio_trust"]).sum() > 0),
        "criterio_muestra_analitica": criterio,
        "n_muestra_analitica": n_m,
        "binarizacion": {
            "confia": list(cfg.TRUST_VALORES_SI),
            "no_confia": list(cfg.TRUST_VALORES_NO),
            "tratado_como_missing": list(cfg.TRUST_VALORES_MISSING),
            "justificacion": "el 5 es 'no se / prefiero no responder': es no-respuesta, "
                             "no desconfianza",
        },
        "missing_por_item": resumen_missing,
        "empleo_es_multirespuesta": True,
        "tabla1": t1.to_dict("records"),
        "confianza": tr.to_dict("records"),
    }, "s06_icare", log)

    log.info("ETAPA 06 COMPLETADA.")


if __name__ == "__main__":
    main()
