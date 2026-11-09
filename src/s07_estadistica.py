# -*- coding: utf-8 -*-
"""
ETAPA 07 — Inferencia estadistica y concordancia entre fuentes.

Que resuelve del informe del asesor
-----------------------------------
* Correcciones #16 y #17 / seccion 2.4: "Para Tabla 3, anadir chi2, gl, p
  y Cramer's V. Con un N grande, interpretar tamano de efecto, no solo
  significancia." y "Reportar IC95% para proporciones clave cuando sea
  posible."
* Seccion 2.3.8 y correccion #19/#20 (Figura 5): "Definir la integracion
  como comparacion a nivel de CATEGORIAS. No existe emparejamiento
  individual entre participantes iCARE y usuarios de Twitter."

Unidad de analisis del contraste principal
------------------------------------------
  LA MENCION, no el tuit ni el autor.

  La version 1 cruzaba polaridad contra la entidad del AUTOR, lo que mide
  el tono de lo que esas cuentas ESCRIBEN, no el tono de lo que se dice
  SOBRE ellas. Como el manuscrito habla de "iCARE entities cited in
  Twitter/X", la unidad correcta es la mencion. Las cifras de la Tabla 3
  cambian respecto del manuscrito anterior: es una correccion.

  Se producen tres versiones del cruce:
    A. PRINCIPAL     una fila por mencion valida.
    B. SENSIBILIDAD  solo tuits que mencionan UNA sola categoria iCARE.
                     Ahi atribuir la polaridad global del tuit a esa
                     categoria es mucho menos discutible, y es la
                     respuesta practica a la correccion #6.
    C. SECUNDARIA    por entidad del autor, etiquetada como lo que es:
                     tono de los mensajes emitidos por cada tipo de cuenta.

Sobre el p-valor con N grande
-----------------------------
Con cientos de miles de menciones cualquier desviacion minima sale
significativa. Por eso se reporta V de Cramer (con correccion de sesgo) y
se interpreta con los umbrales de Cohen ajustados a los grados de
libertad. La lectura del manuscrito debe apoyarse en la V, no en el p.

Concordancia entre fuentes (base de la Figura 5)
------------------------------------------------
Se calcula la correlacion de rangos de Spearman entre:
   x = % de desconfianza declarada (iCARE, n~125, sep-dic 2020)
   y = % de menciones negativas    (Twitter, mar-ago 2020)
sobre las categorias que existen en AMBAS fuentes.

  ADVERTENCIAS que deben acompanar a la figura y al texto:
  - Son DOS POBLACIONES distintas, DOS UNIDADES de analisis distintas
    (personas vs. menciones) y DOS PERIODOS distintos.
  - n = 8 categorias: el coeficiente es descriptivo y su intervalo de
    confianza es muy ancho. No es una prueba de hipotesis.
  - Es concordancia ECOLOGICA. No implica asociacion individual ni
    causalidad, en ninguna direccion.

Salida
------
  resultados/tablas/s07_tabla3_entidad_polaridad_menciones.csv
  resultados/tablas/s07_tabla3_sensibilidad_una_categoria.csv
  resultados/tablas/s07_entidad_polaridad_por_autor.csv
  resultados/tablas/s07_negatividad_por_entidad_ic95.csv
  resultados/tablas/s07_posthoc_residuos.csv
  resultados/tablas/s07_concordancia_cross_source.csv
  resultados/metricas/s07_estadistica.json
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

ETAPA = "s07"
POLARIDADES = ["Negative", "Neutral", "Positive"]


def tabla_contingencia(df, col_cat, col_pol, categorias):
    t = pd.crosstab(df[col_cat], df[col_pol])
    t = t.reindex(index=[c for c in categorias if c in t.index],
                  columns=[p for p in POLARIDADES if p in t.columns], fill_value=0)
    return t


def bloque_prueba(t: pd.DataFrame, nombre: str, log) -> dict:
    """chi2 + V de Cramer + interpretacion, sobre filas/columnas no vacias."""
    t = t.loc[t.sum(axis=1) > 0, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        log.warning("%s: tabla degenerada (%s), no se aplica chi2", nombre, t.shape)
        return {"error": "tabla degenerada", "dimensiones": list(t.shape)}
    r = comun.cramers_v(t.values)
    r["tabla"] = f"{t.shape[0]}x{t.shape[1]}"
    log.info("%s: chi2(%d) = %.1f, p = %.3g, N = %s, V de Cramer = %.4f "
             "(corregida %.4f, efecto %s)",
             nombre, r["gl"], r["chi2"], r["p_valor"], f"{r['n']:,}",
             r["cramers_v"], r["cramers_v_corregida"], r["interpretacion_v"])
    if r["frecuencia_esperada_minima"] < 5:
        log.warning("%s: hay celdas con frecuencia esperada < 5 (%.1f). "
                    "El chi2 puede ser poco fiable; considerar agrupar categorias.",
                    nombre, r["frecuencia_esperada_minima"])
    return r


def tabla_formateada(t: pd.DataFrame) -> pd.DataFrame:
    """Frecuencias absolutas y relativas por fila, claramente separadas
    (el asesor pide justamente que no se mezclen)."""
    pct = (t.div(t.sum(axis=1), axis=0) * 100).round(1)
    out = t.copy()
    out.columns = [f"n_{c.lower()}" for c in t.columns]
    for c in t.columns:
        out[f"pct_{c.lower()}"] = pct[c]
    out["n_total"] = t.sum(axis=1)
    out = out.reset_index()
    out.columns = ["categoria"] + list(out.columns[1:])
    out.insert(1, "etiqueta_en", out["categoria"].map(cfg.ETIQUETA_EN).fillna(out["categoria"]))
    return out.sort_values("pct_negative", ascending=False)


def posthoc(t: pd.DataFrame, nombre: str) -> pd.DataFrame:
    from scipy.stats import norm
    res = comun.residuos_estandarizados(t.values)
    n_celdas = res.size
    filas = []
    for i, cat in enumerate(t.index):
        for j, pol in enumerate(t.columns):
            z = res[i, j]
            p = 2 * (1 - norm.cdf(abs(z)))
            filas.append({
                "analisis": nombre,
                "categoria": cat,
                "etiqueta_en": cfg.ETIQUETA_EN.get(cat, cat),
                "polaridad": pol,
                "observado": int(t.values[i, j]),
                "residuo_estandarizado": round(float(z), 3),
                "p_valor": float(p),
                "p_bonferroni": float(min(1.0, p * n_celdas)),
                "significativo_bonferroni": bool(p * n_celdas < cfg.ALFA),
                "direccion": "mas de lo esperado" if z > 0 else "menos de lo esperado",
            })
    return pd.DataFrame(filas)


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
    log.info("Corpus analitico: %s tuits | menciones validas: %s",
             f"{len(tw):,}", f"{len(men):,}")

    # la polaridad vive a nivel de tuit; se propaga a cada mencion del tuit
    men = men.merge(tw[["polaridad"]].reset_index().rename(columns={"index": "fila_tuit"}),
                    on="fila_tuit", how="left")
    men = men[men["polaridad"].notna()]
    log.info("Menciones con polaridad asignada: %s", f"{len(men):,}")

    cats = cfg.CATEGORIAS_TWITTER
    resultados = {}

    # ---------------- A. principal: menciones ----------------
    log.info("")
    log.info("--- A. ANALISIS PRINCIPAL: entidad mencionada x polaridad (unidad = mencion) ---")
    tA = tabla_contingencia(men, "categoria_mencionada", "polaridad", cats)
    fA = tabla_formateada(tA)
    comun.guardar_tabla(fA, "s07_tabla3_entidad_polaridad_menciones", log)
    log.info("\n%s", fA[["etiqueta_en", "n_total", "pct_negative", "pct_neutral",
                         "pct_positive"]].to_string(index=False))
    resultados["principal_menciones"] = bloque_prueba(tA, "A menciones", log)
    ph = posthoc(tA, "A_menciones")

    # ---------------- B. sensibilidad: una sola categoria ----------------
    log.info("")
    log.info("--- B. SENSIBILIDAD: solo tuits que citan UNA sola categoria ---")
    log.info("En esta submuestra atribuir la polaridad global del tuit a la entidad")
    log.info("citada es defendible. Es la mitigacion de la correccion #6.")
    una = tw[tw["n_categorias_mencionadas"] == 1].copy()
    if len(una):
        una["categoria_unica"] = una["categorias_mencionadas"].map(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) else np.nan)
        una = una[una["categoria_unica"].notna()]
        tB = tabla_contingencia(una, "categoria_unica", "polaridad", cats)
        fB = tabla_formateada(tB)
        comun.guardar_tabla(fB, "s07_tabla3_sensibilidad_una_categoria", log)
        log.info("n = %s tuits\n%s", f"{len(una):,}",
                 fB[["etiqueta_en", "n_total", "pct_negative"]].to_string(index=False))
        resultados["sensibilidad_una_categoria"] = bloque_prueba(tB, "B una categoria", log)
        ph = pd.concat([ph, posthoc(tB, "B_una_categoria")], ignore_index=True)
    else:
        log.warning("No hay tuits con exactamente una categoria mencionada.")
        fB = pd.DataFrame()

    # ---------------- C. secundaria: entidad del autor ----------------
    log.info("")
    log.info("--- C. SECUNDARIA: entidad del AUTOR x polaridad ---")
    log.info("Mide el tono de lo que ESCRIBEN esas cuentas, NO el tono hacia ellas.")
    log.info("Etiquetar asi en el manuscrito si se incluye.")
    tC = tabla_contingencia(tw, "entidad_autor", "polaridad", cats)
    fC = tabla_formateada(tC)
    comun.guardar_tabla(fC, "s07_entidad_polaridad_por_autor", log)
    resultados["secundaria_por_autor"] = bloque_prueba(tC, "C por autor", log)

    comun.guardar_tabla(ph.sort_values(["analisis", "residuo_estandarizado"]),
                        "s07_posthoc_residuos", log)
    sig = ph[ph["significativo_bonferroni"] & (ph["analisis"] == "A_menciones")]
    log.info("Post-hoc (A, Bonferroni): %d de %d celdas se apartan de lo esperado",
             len(sig), int((ph["analisis"] == "A_menciones").sum()))
    if len(sig):
        log.info("\n%s", sig.nlargest(8, "residuo_estandarizado")[
            ["etiqueta_en", "polaridad", "observado", "residuo_estandarizado",
             "direccion"]].to_string(index=False))

    # ---------------- negatividad por entidad con IC95% ----------------
    filas = []
    for cat in tA.index:
        neg = int(tA.loc[cat, "Negative"]) if "Negative" in tA.columns else 0
        tot = int(tA.loc[cat].sum())
        f = comun.fila_proporcion(cfg.ETIQUETA_EN.get(cat, cat), neg, tot)
        f["clave"] = cat
        filas.append(f)
    neg_ic = pd.DataFrame(filas).sort_values("pct", ascending=False)
    neg_ic = neg_ic.rename(columns={"n": "n_negativas", "valid_n": "n_menciones",
                                    "pct": "pct_negativas"})
    comun.guardar_tabla(neg_ic, "s07_negatividad_por_entidad_ic95", log)
    log.info("\nNegatividad por entidad con IC95%%:\n%s",
             neg_ic[["categoria", "n_negativas", "n_menciones", "pct_negativas",
                     "ic95_inf", "ic95_sup"]].to_string(index=False))

    # ---------------- concordancia entre fuentes ----------------
    log.info("")
    log.info("--- CONCORDANCIA ECOLOGICA ENTRE FUENTES (base de la Figura 5) ---")
    conc = pd.DataFrame()
    ruta_trust = cfg.DIR_TABLAS / "s06_confianza_por_fuente.csv"
    if not ruta_trust.exists():
        log.warning("Falta s06_confianza_por_fuente.csv: se omite la concordancia. "
                    "Ejecutar la etapa 06.")
    else:
        tr = pd.read_csv(ruta_trust)
        dis = tr[tr["medida"] == "Distrust"].set_index("clave")
        filas = []
        for cat in cfg.CATEGORIAS_CROSS_SOURCE:
            if cat not in dis.index or cat not in neg_ic["clave"].values:
                log.info("  categoria '%s' no comparable (falta en una de las fuentes)", cat)
                continue
            fila_neg = neg_ic[neg_ic["clave"] == cat].iloc[0]
            filas.append({
                "clave": cat,
                "etiqueta_en": cfg.ETIQUETA_EN[cat],
                "icare_pct_distrust": float(dis.loc[cat, "pct"]),
                "icare_ic95_inf": float(dis.loc[cat, "ic95_inf"]),
                "icare_ic95_sup": float(dis.loc[cat, "ic95_sup"]),
                "icare_valid_n": int(dis.loc[cat, "valid_n"]),
                "twitter_pct_negative": float(fila_neg["pct_negativas"]),
                "twitter_ic95_inf": float(fila_neg["ic95_inf"]),
                "twitter_ic95_sup": float(fila_neg["ic95_sup"]),
                "twitter_n_menciones": int(fila_neg["n_menciones"]),
            })
        conc = pd.DataFrame(filas)
        if len(conc) >= 3:
            from scipy.stats import spearmanr
            conc["rango_icare"] = conc["icare_pct_distrust"].rank(ascending=False).astype(int)
            conc["rango_twitter"] = conc["twitter_pct_negative"].rank(ascending=False).astype(int)
            conc["diferencia_de_rango"] = conc["rango_icare"] - conc["rango_twitter"]
            rho, pval = spearmanr(conc["icare_pct_distrust"], conc["twitter_pct_negative"])

            rng = np.random.default_rng(cfg.SEMILLA)
            idx = np.arange(len(conc))
            boot = []
            for _ in range(cfg.BOOTSTRAP_N):
                s = rng.choice(idx, size=len(idx), replace=True)
                if len(np.unique(s)) < 3:
                    continue
                r, _ = spearmanr(conc["icare_pct_distrust"].values[s],
                                 conc["twitter_pct_negative"].values[s])
                if not np.isnan(r):
                    boot.append(r)
            lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) \
                if boot else (np.nan, np.nan)

            conc = conc.sort_values("icare_pct_distrust", ascending=False)
            comun.guardar_tabla(conc, "s07_concordancia_cross_source", log)
            log.info("\n%s", conc[["etiqueta_en", "icare_pct_distrust", "twitter_pct_negative",
                                   "rango_icare", "rango_twitter",
                                   "diferencia_de_rango"]].to_string(index=False))
            log.info("Spearman rho = %.3f (p = %.3g), IC95%% bootstrap [%.3f, %.3f], "
                     "sobre %d categorias", rho, pval, lo, hi, len(conc))
            log.info("LECTURA OBLIGADA: dos poblaciones, dos unidades de analisis y dos")
            log.info("periodos distintos. Es concordancia ecologica descriptiva. NO es")
            log.info("evidencia de asociacion individual ni de causalidad.")
            resultados["concordancia"] = {
                "spearman_rho": float(rho), "p_valor": float(pval),
                "ic95_bootstrap": [lo, hi], "n_categorias": int(len(conc)),
                "n_bootstrap": len(boot),
                "categorias_excluidas": [
                    c for c in cfg.CATEGORIAS
                    if c not in conc["clave"].values],
                "advertencias": [
                    "dos poblaciones distintas (participantes iCARE vs. usuarios de Twitter)",
                    "dos unidades de analisis (personas vs. menciones)",
                    "dos periodos (sep-dic 2020 vs. mar-ago 2020)",
                    "n = %d categorias: coeficiente descriptivo, no prueba de hipotesis" % len(conc),
                    "concordancia ecologica: no implica asociacion individual ni causalidad",
                ],
            }
        else:
            log.warning("Menos de 3 categorias comparables: no se calcula la concordancia.")

    comun.guardar_metricas({
        "unidad_analisis_principal": "mencion valida",
        "cambio_respecto_v1": ("la version anterior cruzaba polaridad contra la entidad del "
                               "AUTOR; el analisis principal ahora usa la entidad MENCIONADA"),
        "polaridad_es_target_dependent": cfg.POLARIDAD_ES_TARGET_DEPENDENT,
        "n_menciones": int(len(men)),
        "n_tuits": int(len(tw)),
        "pruebas": resultados,
        "negatividad_por_entidad": neg_ic.to_dict("records"),
        "concordancia_tabla": conc.to_dict("records") if len(conc) else [],
        "nota_p_valor": ("con N grande el p-valor es casi siempre significativo; la lectura "
                         "debe apoyarse en la V de Cramer"),
    }, "s07_estadistica", log)

    log.info("ETAPA 07 COMPLETADA.")


if __name__ == "__main__":
    main()
