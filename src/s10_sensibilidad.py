# -*- coding: utf-8 -*-
"""
ETAPA 10 - Analisis de sensibilidad de alta precision.

Que resuelve del informe del asesor (segunda ronda, Q2/Q3)
---------------------------------------------------------
* Correccion 1.3 "Realizar analisis de sensibilidad con un corpus
  colombiano de alta precision": la regla de MAYORIA (>= 2 de 3
  evidencias) con la que se construye el corpus principal tiene una
  precision validada de 0.740 contando los indeterminados como error y
  de 0.794 excluyendolos. La regla de UNANIMIDAD (3 de 3) la sube a
  0.938 a costa de conservar el 35.3% del corpus. Aqui se repite el
  resultado principal sobre ese subcorpus.
* Correccion 1.4 "Realizar analisis de sensibilidad con clasificacion de
  entidades de alta precision": la precision de asignacion de categoria
  a nivel de mencion va de 0.93 (medios) a 0.63 (lider nacional) en las
  cuatro categorias interpretadas, y el componente debil es el zero-shot.
  Aqui se repite el resultado usando SOLO las menciones resueltas por el
  diccionario curado, es decir sin ninguna contribucion del zero-shot.

Criterio de exito declarado por el asesor
-----------------------------------------
Si el orden sustantivo de las categorias y la magnitud del efecto se
mantienen, es evidencia de robustez y asi debe reportarse. Si cambian de
forma importante, hay que ajustar la interpretacion principal. Este
script no decide por el manuscrito: imprime las tres versiones una al
lado de la otra y deja el juicio a la vista.

Lo que NO hace
--------------
No sustituye a la etapa 07. El analisis principal sigue siendo el de la
etapa 07 sobre el corpus completo, porque es el mayor y porque las cuatro
categorias interpretadas ya superan el umbral de precision. Esto son
comprobaciones de robustez.

ENTRADA
-------
  intermedios/s05_corpus_analitico.parquet   (polaridad + n_votos_colombia)
  intermedios/s04_menciones.parquet          (categoria y origen de cada mencion)
  resultados/tablas/s06_confianza_por_fuente.csv
  resultados/tablas/s09_precision_mencion_por_categoria.csv
  resultados/validacion/anotados/*.xlsx      (para reestimar la precision
                                              geografica bajo unanimidad)

SALIDA
------
  resultados/tablas/s10_sensibilidad_entidad_polaridad.csv
  resultados/tablas/s10_tablaA2_comparacion_corpus.csv   <- Tabla A2 del paper
  resultados/tablas/s10_concordancia_por_corpus.csv
  resultados/tablas/s10_precision_geo_por_regla.csv
  resultados/metricas/s10_sensibilidad.json

USO
---
  python src/s10_sensibilidad.py
  python correr.py 10
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

ETAPA = "s10"
POLARIDADES = ["Negative", "Neutral", "Positive"]

# Etiqueta con la que cada corpus aparece en la Tabla A2 del manuscrito.
CORPUS = {
    "principal": "Main corpus",
    "geo_unanime": "High-precision geographic corpus",
    "dicc_curado": "Curated-dictionary corpus",
    "ambos": "Both restrictions",
}

# Valor de origen_mencion que marca las menciones resueltas a mano.
ORIGEN_DICCIONARIO = "R1_diccionario_curado"


# ---------------------------------------------------------------------------
# CATEGORIAS INTERPRETABLES
# ---------------------------------------------------------------------------
def categorias_interpretadas(log) -> list[str]:
    """Las categorias que superan el umbral de precision de la etapa 09.

    Se leen del CSV de la etapa 09 y no se escriben a mano: si cambia la
    anotacion, cambia el umbral o cambia la precision, esta lista cambia
    sola y el analisis de sensibilidad sigue hablando de las mismas
    categorias que Results.
    """
    d = comun_leer("s09_precision_mencion_por_categoria")
    if d is None or "analizable" not in d.columns:
        log.warning("Falta s09_precision_mencion_por_categoria.csv: se usan "
                    "todas las categorias observables en Twitter/X. Correr "
                    "antes la etapa 09 para restringir a las validadas.")
        return list(cfg.CATEGORIAS_TWITTER)
    ok = list(d.loc[d["analizable"].astype(bool), "categoria"])
    log.info("Categorias interpretadas (precision de mencion >= umbral de la "
             "etapa 09): %s", ", ".join(ok))
    return ok


def comun_leer(nombre: str):
    ruta = cfg.DIR_TABLAS / f"{nombre}.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


# ---------------------------------------------------------------------------
# PRECISION GEOGRAFICA POR REGLA
# ---------------------------------------------------------------------------
def precision_geo_por_regla(log) -> pd.DataFrame:
    """Reestima la precision del corpus colombiano bajo mayoria y unanimidad.

    La muestra de validacion geografica conserva n_votos_colombia, asi que
    la precision de la regla de unanimidad se puede calcular sobre los
    mismos 400 casos anotados a mano, sin volver a anotar nada. Es el
    numero que el manuscrito cita como 0.938.
    """
    try:
        import s09_evaluar_anotacion as s09
    except Exception as e:            # pragma: no cover
        log.warning("No se pudo importar la etapa 09 (%s): se omite la "
                    "reestimacion de la precision geografica.", e)
        return pd.DataFrame()

    d = s09.cargar_tarea("origen_geo", log)
    if d is None:
        return pd.DataFrame()
    adj = s09.cargar_adjudicacion(log)
    d = s09.construir_referencia(d, "origen_geo", adj, log)

    muestra = pd.read_excel(cfg.DIR_VALIDACION / "muestra_origen_geo.xlsx",
                            sheet_name="muestra")[["id_muestra", "n_votos_colombia"]]
    d = d.merge(muestra.rename(columns={"id_muestra": "id"}), on="id", how="left")

    filas = []
    for etiqueta, sub in (("mayoria (>= 2 de 3)", d),
                          ("unanimidad (3 de 3)", d[d["n_votos_colombia"] == 3])):
        n = len(sub)
        gold = sub["gold"].astype(str).str.strip().str.lower()
        sin_juicio = gold.isin(s09.NO_ES_CLASE)
        aciertos = int((gold == "si").sum())
        # dos lecturas, como en el manuscrito: contando los indeterminados
        # como error y excluyendolos del denominador
        p1, lo1, hi1 = comun.ic_wilson(aciertos, n)
        n2 = int(n - sin_juicio.sum())
        p2, lo2, hi2 = comun.ic_wilson(aciertos, n2)
        filas.append({
            "regla": etiqueta,
            "n_muestra": n,
            "n_colombianos_en_referencia": aciertos,
            "n_sin_juicio": int(sin_juicio.sum()),
            "precision": round(p1, 4),
            "ic95_inf": round(lo1, 4),
            "ic95_sup": round(hi1, 4),
            "precision_excl_sin_juicio": round(p2, 4),
            "ic95_inf_excl": round(lo2, 4),
            "ic95_sup_excl": round(hi2, 4),
        })
    out = pd.DataFrame(filas)
    log.info("\nPrecision del corpus colombiano por regla de decision:\n%s",
             out[["regla", "n_muestra", "precision", "ic95_inf", "ic95_sup",
                  "precision_excl_sin_juicio"]].to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# ANALISIS POR CORPUS
# ---------------------------------------------------------------------------
def analizar(men: pd.DataFrame, cats: list[str], clave: str, log) -> dict:
    """Cruce entidad x polaridad, IC95% de negatividad y concordancia."""
    sub = men[men["categoria_mencionada"].isin(cats)]
    t = pd.crosstab(sub["categoria_mencionada"], sub["polaridad"])
    t = t.reindex(index=[c for c in cats if c in t.index],
                  columns=[p for p in POLARIDADES if p in t.columns], fill_value=0)
    if t.shape[0] < 2 or t.shape[1] < 2:
        log.warning("%s: tabla degenerada, se omite.", clave)
        return {}

    prueba = comun.cramers_v(t.values)
    filas = []
    for cat in t.index:
        neg = int(t.loc[cat, "Negative"])
        tot = int(t.loc[cat].sum())
        p, lo, hi = comun.ic_wilson(neg, tot)
        filas.append({
            "corpus": CORPUS[clave],
            "clave_corpus": clave,
            "categoria": cat,
            "etiqueta_en": cfg.ETIQUETA_EN.get(cat, cat),
            "n_negative": neg,
            "n_neutral": int(t.loc[cat, "Neutral"]),
            "n_positive": int(t.loc[cat, "Positive"]),
            "n_total": tot,
            "pct_negative": round(p * 100, 1),
            "ic95_inf": round(lo * 100, 1),
            "ic95_sup": round(hi * 100, 1),
        })
    d = pd.DataFrame(filas).sort_values("pct_negative", ascending=False)
    d["rango_negatividad"] = range(1, len(d) + 1)

    log.info("\n--- %s ---", CORPUS[clave])
    log.info("\n%s", d[["etiqueta_en", "n_total", "pct_negative", "ic95_inf",
                        "ic95_sup"]].to_string(index=False))
    log.info("chi2(%d) = %.1f, p = %.3g, N = %s, V de Cramer = %.3f "
             "(corregida %.3f, efecto %s)",
             prueba["gl"], prueba["chi2"], prueba["p_valor"], f"{prueba['n']:,}",
             prueba["cramers_v"], prueba["cramers_v_corregida"],
             prueba["interpretacion_v"])
    return {"tabla": d, "prueba": prueba}


def concordancia(d: pd.DataFrame, dis: pd.DataFrame, clave: str, log) -> pd.DataFrame:
    """Concordancia ecologica entre desconfianza declarada y tono negativo.

    Las advertencias de la etapa 07 siguen valiendo aqui sin cambios: dos
    poblaciones, dos unidades de analisis y dos periodos distintos; con
    n = 3 categorias el coeficiente es descriptivo y no una prueba.
    """
    from scipy.stats import spearmanr
    filas = []
    for cat in cfg.CATEGORIAS_CROSS_SOURCE:
        if cat not in dis.index or cat not in set(d["categoria"]):
            continue
        f = d[d["categoria"] == cat].iloc[0]
        filas.append({
            "corpus": CORPUS[clave],
            "clave_corpus": clave,
            "categoria": cat,
            "etiqueta_en": cfg.ETIQUETA_EN[cat],
            "icare_pct_distrust": float(dis.loc[cat, "pct"]),
            "icare_valid_n": int(dis.loc[cat, "valid_n"]),
            "twitter_pct_negative": float(f["pct_negative"]),
            "twitter_n_menciones": int(f["n_total"]),
        })
    c = pd.DataFrame(filas)
    if len(c) < 3:
        log.warning("%s: menos de 3 categorias comparables, no hay concordancia.", clave)
        return c
    c["rango_icare"] = c["icare_pct_distrust"].rank(ascending=False).astype(int)
    c["rango_twitter"] = c["twitter_pct_negative"].rank(ascending=False).astype(int)
    rho, p = spearmanr(c["icare_pct_distrust"], c["twitter_pct_negative"])
    c["spearman_rho"] = round(float(rho), 3)
    c["p_valor"] = round(float(p), 3)
    log.info("Concordancia (%s): rho = %.3f sobre %d categorias%s",
             CORPUS[clave], rho, len(c),
             "  [ordenes identicos]" if (c["rango_icare"] == c["rango_twitter"]).all()
             else "  [ORDENES DISTINTOS: revisar la interpretacion]")
    return c


# ---------------------------------------------------------------------------
def main() -> None:
    p = comun.parser_base(__doc__)
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    for req in (cfg.PQ_ANALITICO, cfg.PQ_MENCIONES):
        if not req.exists():
            log.error("Falta %s. Ejecutar las etapas 04 y 05.", req.name)
            sys.exit(1)

    tw = pd.read_parquet(cfg.PQ_ANALITICO)
    men = pd.read_parquet(cfg.PQ_MENCIONES)
    tw = tw.reset_index().rename(columns={"index": "fila_tuit"})
    men = men.merge(tw[["fila_tuit", "polaridad", "n_votos_colombia"]],
                    on="fila_tuit", how="left")
    men = men[men["polaridad"].notna()]

    n_unan = int((tw["n_votos_colombia"] == 3).sum())
    log.info("Corpus principal: %s tuits, %s menciones con polaridad.",
             f"{len(tw):,}", f"{len(men):,}")
    log.info("Corpus geografico de alta precision (unanimidad): %s tuits "
             "(%.1f%% del corpus principal).", f"{n_unan:,}", 100 * n_unan / len(tw))
    n_dicc = int((men["origen_mencion"] == ORIGEN_DICCIONARIO).sum())
    log.info("Menciones resueltas por diccionario curado: %s (%.1f%% de las validas).",
             f"{n_dicc:,}", 100 * n_dicc / len(men))

    cats = categorias_interpretadas(log)

    # precision geografica reestimada bajo cada regla (numero 0.938 del paper)
    prec_geo = precision_geo_por_regla(log)
    if len(prec_geo):
        comun.guardar_tabla(prec_geo, "s10_precision_geo_por_regla", log)

    subconjuntos = {
        "principal": men,
        "geo_unanime": men[men["n_votos_colombia"] == 3],
        "dicc_curado": men[men["origen_mencion"] == ORIGEN_DICCIONARIO],
        "ambos": men[(men["n_votos_colombia"] == 3)
                     & (men["origen_mencion"] == ORIGEN_DICCIONARIO)],
    }

    tablas, concord, pruebas = [], [], {}
    for clave, sub in subconjuntos.items():
        r = analizar(sub, cats, clave, log)
        if not r:
            continue
        tablas.append(r["tabla"])
        pruebas[clave] = r["prueba"]

    largo = pd.concat(tablas, ignore_index=True)
    comun.guardar_tabla(largo, "s10_sensibilidad_entidad_polaridad", log)

    # concordancia por corpus
    ruta_trust = cfg.DIR_TABLAS / "s06_confianza_por_fuente.csv"
    if ruta_trust.exists():
        tr = pd.read_csv(ruta_trust)
        dis = tr[tr["medida"] == "Distrust"].set_index("clave")
        for clave in subconjuntos:
            d = largo[largo["clave_corpus"] == clave]
            if len(d):
                c = concordancia(d, dis, clave, log)
                if len(c):
                    concord.append(c)
        if concord:
            comun.guardar_tabla(pd.concat(concord, ignore_index=True),
                                "s10_concordancia_por_corpus", log)
    else:
        log.warning("Falta s06_confianza_por_fuente.csv: se omite la concordancia. "
                    "Ejecutar la etapa 06.")

    # ---------------- Tabla A2 del manuscrito ----------------
    orden = [c for c in largo[largo["clave_corpus"] == "principal"]
             .sort_values("pct_negative", ascending=False)["categoria"]]
    columnas = ["principal", "geo_unanime", "dicc_curado"]
    filas = []
    for cat in orden:
        fila = {"iCARE entity category": cfg.ETIQUETA_EN.get(cat, cat)}
        for clave in columnas:
            f = largo[(largo["clave_corpus"] == clave) & (largo["categoria"] == cat)]
            fila[CORPUS[clave]] = (f"{f.iloc[0]['pct_negative']:.1f} "
                                   f"({f.iloc[0]['n_total']:,})") if len(f) else "-"
        filas.append(fila)
    for etiqueta, extraer in (
            ("chi2(gl)", lambda r: f"{r['chi2']:.1f}"),
            ("Cramer's V (bias-corrected)",
             lambda r: f"{r['cramers_v']:.3f} ({r['cramers_v_corregida']:.3f})"),
            ("n (mentions)", lambda r: f"{r['n']:,}"),
            ("p", lambda r: "< 0.001" if r["p_valor"] < 0.001 else f"{r['p_valor']:.3g}")):
        fila = {"iCARE entity category": etiqueta}
        for clave in columnas:
            fila[CORPUS[clave]] = extraer(pruebas[clave]) if clave in pruebas else "-"
        filas.append(fila)
    tabla_a2 = pd.DataFrame(filas)
    comun.guardar_tabla(tabla_a2, "s10_tablaA2_comparacion_corpus", log)
    log.info("\nTABLA A2 (copiar al manuscrito):\n%s", tabla_a2.to_string(index=False))

    # ---------------- lectura del criterio de exito ----------------
    ordenes = {clave: list(largo[largo["clave_corpus"] == clave]
                           .sort_values("pct_negative", ascending=False)["categoria"])
               for clave in subconjuntos if clave in pruebas}
    igual = all(o == ordenes["principal"] for o in ordenes.values())
    log.info("")
    if igual:
        log.info("ROBUSTEZ: el orden de las categorias es identico en los %d "
                 "corpus. Reportable como evidencia de robustez.", len(ordenes))
    else:
        log.warning("El orden de las categorias NO es identico entre corpus. "
                    "Comparar los IC95% antes de concluir: un intercambio "
                    "entre categorias con intervalos solapados no es un "
                    "cambio sustantivo, pero debe declararse en Results.")
        for clave, o in ordenes.items():
            log.warning("  %-34s %s", CORPUS[clave],
                        " > ".join(cfg.ETIQUETA_EN.get(c, c) for c in o))

    comun.guardar_metricas({
        "proposito": ("comprobaciones de robustez de las correcciones 1.3 y 1.4 "
                      "de la segunda ronda; el analisis principal sigue siendo "
                      "el de la etapa 07"),
        "categorias_interpretadas": cats,
        "n_tuits_corpus_principal": int(len(tw)),
        "n_tuits_unanimes": n_unan,
        "pct_tuits_unanimes": round(100 * n_unan / len(tw), 2),
        "n_menciones_validas": int(len(men)),
        "n_menciones_diccionario_curado": n_dicc,
        "pct_menciones_diccionario_curado": round(100 * n_dicc / len(men), 2),
        "pruebas": pruebas,
        "orden_por_corpus": ordenes,
        "orden_identico_entre_corpus": bool(igual),
        "precision_geo_por_regla": prec_geo.to_dict("records") if len(prec_geo) else [],
        "concordancia": (pd.concat(concord, ignore_index=True).to_dict("records")
                         if concord else []),
        "advertencias": [
            "la concordancia sigue siendo ecologica y descriptiva en todos los corpus",
            "n = 3 categorias comparables: no hay soporte inferencial",
            "restringir el corpus reduce la potencia; los IC95% se ensanchan",
        ],
    }, "s10_sensibilidad", log)

    log.info("ETAPA 10 COMPLETADA.")


if __name__ == "__main__":
    main()
