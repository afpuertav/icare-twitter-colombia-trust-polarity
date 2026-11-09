# -*- coding: utf-8 -*-
"""
ETAPA 09 - Evaluacion de la anotacion humana.

Calcula, a partir de los Excel anotados a mano, todo lo que el paper
reporta sobre la fiabilidad de los clasificadores automaticos:

  1. Acuerdo interanotador: acuerdo bruto y kappa de Cohen por tarea.
     -> cierra la parte "acuerdo interanotador" de la correccion #8.
  2. Precision del modelo por clase predicha, con IC95% de Wilson.
     -> cierra las correcciones #5 (modelo de polaridad) y #7 (geo).
  3. Reponderacion al corpus: exactitud global y prevalencia corregida.
  4. Precision de la categoria a nivel de MENCION, combinando el
     diccionario curado con la precision validada del zero-shot, y que
     categorias superan el umbral para poder analizarse.

Todo sale de este script. Ningun numero del manuscrito debe escribirse
a mano: si cambian las anotaciones, se vuelve a correr y se copian las
cifras de las tablas que genera.

ENTRADA (carpeta resultados/validacion/anotados/)
------------------------------------------------
  1_polaridad_anotado.xlsx          2_entidad_anotado.xlsx          3_origen_colombiano_anotado.xlsx
  1_polaridad_anotado2.xlsx         2_entidad_anotado2.xlsx         3_origen_colombiano_anotado2.xlsx
  ADJUDICAR_desacuerdos.xlsx   (opcional; si la columna DECISION_FINAL
                                esta rellena, esas decisiones mandan)

Cada archivo tiene una hoja 'anotar' con las columnas id y anotacion.
El sufijo "2" es el segundo anotador. El orden de las filas es
irrelevante: todo se cruza por 'id'.

COMO SE CONSTRUYE EL ESTANDAR DE REFERENCIA
-------------------------------------------
  - Si los dos anotadores coinciden -> esa es la etiqueta de referencia.
  - Si discrepan y hay decision adjudicada -> manda la adjudicacion.
  - Si discrepan y no hay adjudicacion -> el caso se EXCLUYE y se reporta
    cuantos se excluyeron. Ojo: excluir desacuerdos mide solo sobre los
    casos faciles y sobreestima la precision, asi que conviene adjudicar.

USO
---
  python src/s09_evaluar_anotacion.py
  python src/s09_evaluar_anotacion.py --tarea entidad
  python src/s09_evaluar_anotacion.py --sin-adjudicacion   (ignora el archivo adjudicado)

SALIDA
------
  resultados/tablas/s09_acuerdo_interanotador.csv
  resultados/tablas/s09_precision_<tarea>.csv
  resultados/tablas/s09_matriz_<tarea>.csv
  resultados/tablas/s09_polaridad_corregida.csv
  resultados/tablas/s09_precision_mencion_por_categoria.csv
  resultados/metricas/s09_validacion_humana.json
  resultados/validacion/s09_calculo_auditable.xlsx
        <- libro de Excel con las matrices de confusion y el kappa
           calculado con FORMULAS de Excel, para poder auditar la cuenta
           a mano y rehacerla sin Python.
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

ETAPA = "s09"

# ---------------------------------------------------------------------------
# PARAMETROS EDITABLES
# ---------------------------------------------------------------------------

DIR_ANOTADOS = cfg.DIR_VALIDACION / "anotados"
ARCHIVO_ADJUDICACION = DIR_ANOTADOS / "ADJUDICAR_desacuerdos.xlsx"

# Umbral de precision de mencion por debajo del cual una categoria no se
# interpreta en Results. Cambiarlo aqui cambia la tabla de decision.
UMBRAL_PRECISION_ANALIZABLE = 0.55

# tarea -> (archivo anotador 1, archivo anotador 2, muestra original con la
#           prediccion del modelo, nombre de la columna de prediccion)
TAREAS = {
    "polaridad": ("1_polaridad_anotado.xlsx", "1_polaridad_anotado2.xlsx",
                  "muestra_polaridad.xlsx"),
    "entidad": ("2_entidad_anotado.xlsx", "2_entidad_anotado2.xlsx",
                "muestra_entidad.xlsx"),
    "origen_geo": ("3_origen_colombiano_anotado.xlsx", "3_origen_colombiano_anotado2.xlsx",
                   "muestra_origen_geo.xlsx"),
}

# En origen geografico el modelo dice "colombiano" para los 400 casos y el
# anotador responde si/no/no_se. Este mapa traduce para poder comparar.
EQUIVALENCIA_GEO = {"colombiano": "si"}

# Etiquetas que no son una clase, sino ausencia de juicio.
NO_ES_CLASE = {"no_se", "descartar", "", "nan", "none"}

# Distribucion de las predicciones de polaridad en el corpus completo.
# Se lee de s05_distribucion_polaridad.csv; esto es solo el respaldo.
PESOS_POLARIDAD_RESPALDO = {"Negative": 19950, "Neutral": 26504, "Positive": 5349}


# ---------------------------------------------------------------------------
# CARGA
# ---------------------------------------------------------------------------

def _wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC95% de Wilson. Implementado aqui para que la etapa 09 no dependa de
    scipy y se pueda correr en cualquier maquina; coincide con
    comun.ic_wilson al 95%."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = exitos / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    medio = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centro - medio, centro + medio)


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def cargar_tarea(tarea: str, log) -> pd.DataFrame | None:
    """Devuelve id | A | B | prediccion_modelo, o None si faltan archivos."""
    f1, f2, orig = TAREAS[tarea]
    p1, p2, po = DIR_ANOTADOS / f1, DIR_ANOTADOS / f2, cfg.DIR_VALIDACION / orig
    if not p1.exists():
        log.warning("Falta %s; se omite la tarea %s.", p1.name, tarea)
        return None
    a = pd.read_excel(p1, sheet_name="anotar")[["id", "anotacion"]].rename(columns={"anotacion": "A"})
    if p2.exists():
        b = pd.read_excel(p2, sheet_name="anotar")[["id", "anotacion"]].rename(columns={"anotacion": "B"})
        d = a.merge(b, on="id", how="outer")
    else:
        log.warning("No hay segundo anotador para %s: no habra kappa.", tarea)
        d = a.copy()
        d["B"] = np.nan
    if not po.exists():
        log.error("Falta la muestra original %s.", po.name)
        return None
    m = pd.read_excel(po, sheet_name="muestra")[["id_muestra", "prediccion_modelo"]]
    m = m.rename(columns={"id_muestra": "id"})
    d = d.merge(m, on="id", how="left")
    for c in ("A", "B", "prediccion_modelo"):
        d[c] = _norm(d[c])
    if tarea == "origen_geo":
        d["prediccion_modelo"] = d["prediccion_modelo"].str.lower().map(
            lambda x: EQUIVALENCIA_GEO.get(x, x))
    return d


def cargar_adjudicacion(log) -> dict[tuple[str, int], str]:
    """(tarea, id) -> decision final. Vacio si no hay archivo o esta sin rellenar."""
    if not ARCHIVO_ADJUDICACION.exists():
        return {}
    d = pd.read_excel(ARCHIVO_ADJUDICACION, sheet_name="adjudicar")
    if "DECISION_FINAL" not in d.columns:
        log.warning("%s no tiene columna DECISION_FINAL.", ARCHIVO_ADJUDICACION.name)
        return {}
    d = d[d["DECISION_FINAL"].notna()]
    d["DECISION_FINAL"] = _norm(d["DECISION_FINAL"])
    d = d[~d["DECISION_FINAL"].str.lower().isin({"", "nan"})]
    salida = {(str(r["tarea"]).strip(), int(r["id"])): r["DECISION_FINAL"] for _, r in d.iterrows()}
    log.info("Adjudicacion: %s decisiones cargadas de %s.", len(salida), ARCHIVO_ADJUDICACION.name)
    return salida


def construir_referencia(d: pd.DataFrame, tarea: str, adj: dict, log) -> pd.DataFrame:
    """Anade la columna 'gold' y 'origen_gold'."""
    d = d.copy()
    clave_tarea = {"polaridad": "polaridad", "entidad": "entidad", "origen_geo": "geo"}[tarea]
    gold, origen = [], []
    for _, r in d.iterrows():
        a, b = r["A"], r["B"]
        if b in ("nan", ""):
            gold.append(a); origen.append("anotador_unico")
        elif a == b:
            gold.append(a); origen.append("acuerdo")
        else:
            dec = adj.get((clave_tarea, int(r["id"])))
            if dec:
                gold.append(dec); origen.append("adjudicado")
            else:
                gold.append(np.nan); origen.append("desacuerdo_sin_adjudicar")
    d["gold"] = gold
    d["origen_gold"] = origen
    n_pend = (d["origen_gold"] == "desacuerdo_sin_adjudicar").sum()
    if n_pend:
        log.warning("%s: %s desacuerdos SIN adjudicar quedan fuera del calculo. "
                    "Esto sobreestima la precision; rellena %s.",
                    tarea, n_pend, ARCHIVO_ADJUDICACION.name)
    return d


# ---------------------------------------------------------------------------
# CALCULOS
# ---------------------------------------------------------------------------

def acuerdo_interanotador(d: pd.DataFrame) -> dict:
    """Acuerdo bruto y kappa de Cohen sobre los casos con ambos anotadores."""
    m = d[(d["B"] != "nan") & (d["B"] != "")]
    if m.empty:
        return {"n": 0, "acuerdo_bruto": None, "kappa": None}
    a, b = m["A"].tolist(), m["B"].tolist()
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca = {c: a.count(c) / n for c in cats}
    cb = {c: b.count(c) / n for c in cats}
    pe = sum(ca[c] * cb[c] for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    # error estandar aproximado de Fleiss para el kappa de Cohen
    se = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)) if pe < 1 else float("nan")
    return {"n": n, "acuerdo_bruto": po, "p_esperado": pe, "kappa": kappa,
            "kappa_ic95_inf": kappa - 1.96 * se, "kappa_ic95_sup": kappa + 1.96 * se,
            "n_desacuerdos": int(n - po * n), "interpretacion": interpretar_kappa(kappa)}


def interpretar_kappa(k: float) -> str:
    if k is None or (isinstance(k, float) and np.isnan(k)): return "no calculable"
    if k < 0.20: return "pobre"
    if k < 0.40: return "debil"
    if k < 0.60: return "moderado"
    if k < 0.80: return "sustancial"
    return "casi perfecto"


def precision_por_clase(d: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """Precision del modelo para cada clase que predice, con IC95% Wilson."""
    g = d[d["gold"].notna()].copy()
    filas = []
    for clase in sorted(g["prediccion_modelo"].unique()):
        s = g[g["prediccion_modelo"] == clase]
        # las no-clases (no_se, descartar) no cuentan como acierto ni se excluyen
        # por defecto: se reportan las dos lecturas.
        n_total = len(s)
        aciertos = int((s["gold"] == clase).sum())
        n_juzgables = int((~s["gold"].str.lower().isin(NO_ES_CLASE)).sum())
        lo, hi = _wilson(aciertos, n_total)
        lo2, hi2 = _wilson(aciertos, n_juzgables) if n_juzgables else (np.nan, np.nan)
        filas.append({
            "tarea": tarea, "clase_predicha": clase,
            "n_evaluados": n_total, "n_correctos": aciertos,
            "precision": aciertos / n_total if n_total else np.nan,
            "ic95_inf": lo, "ic95_sup": hi,
            "n_sin_juicio": n_total - n_juzgables,
            "precision_excl_sin_juicio": aciertos / n_juzgables if n_juzgables else np.nan,
            "ic95_inf_excl": lo2, "ic95_sup_excl": hi2,
        })
    return pd.DataFrame(filas)


def matriz_confusion(d: pd.DataFrame) -> pd.DataFrame:
    g = d[d["gold"].notna()]
    return pd.crosstab(g["gold"], g["prediccion_modelo"], dropna=False)


def reponderar_polaridad(d: pd.DataFrame, pesos: dict) -> pd.DataFrame:
    """Exactitud global y prevalencia corregida usando la matriz de confusion."""
    g = d[d["gold"].notna()]
    clases = [c for c in pesos if c in set(g["prediccion_modelo"])]
    ct = pd.crosstab(g["gold"], g["prediccion_modelo"]).reindex(
        index=clases, columns=clases, fill_value=0)
    N = sum(pesos[c] for c in clases)
    filas = []
    for real in clases:
        corr = sum(pesos[c] * (ct.loc[real, c] / ct[c].sum()) for c in clases if ct[c].sum())
        filas.append({"clase": real,
                      "n_modelo": pesos[real], "pct_modelo": 100 * pesos[real] / N,
                      "n_corregido": corr, "pct_corregido": 100 * corr / N,
                      "factor": corr / pesos[real] if pesos[real] else np.nan})
    exact = sum(pesos[c] * (ct.loc[c, c] / ct[c].sum()) for c in clases if ct[c].sum()) / N
    out = pd.DataFrame(filas)
    out.attrs["exactitud_reponderada"] = exact
    return out


def precision_mencion(prec_entidad: pd.DataFrame, log) -> pd.DataFrame:
    """Combina el diccionario curado (precision ~1) con el zero-shot validado."""
    ruta = cfg.DIR_TABLAS / "s04_resolucion_menciones_por_categoria.csv"
    if not ruta.exists():
        log.warning("Falta %s; no se calcula la precision de mencion. "
                    "Se genera al correr s04 o desde s04_menciones.parquet.", ruta.name)
        return pd.DataFrame()
    res = pd.read_csv(ruta)
    p = prec_entidad.set_index("clase_predicha")["precision"].to_dict()
    filas = []
    for _, r in res.iterrows():
        cat = r["categoria"]
        if cat not in p:
            continue
        f = r["pct_curado"] / 100.0
        pm = f + (1 - f) * p[cat]
        filas.append({"categoria": cat, "n_menciones": int(r["n_menciones"]),
                      "pct_curado": r["pct_curado"], "precision_zero_shot": p[cat],
                      "precision_mencion": pm,
                      "analizable": bool(pm >= UMBRAL_PRECISION_ANALIZABLE)})
    out = pd.DataFrame(filas).sort_values("n_menciones", ascending=False)
    return out


# ---------------------------------------------------------------------------
# LIBRO DE EXCEL AUDITABLE
# ---------------------------------------------------------------------------

def exportar_auditable(por_tarea: dict, ruta: Path, log) -> None:
    """Escribe las matrices de confusion A x B con el kappa en FORMULAS de Excel."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        log.warning("openpyxl no disponible; no se escribe el libro auditable.")
        return
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    neg = Font(bold=True)
    cab = PatternFill("solid", fgColor="44546A")
    am = PatternFill("solid", fgColor="FFF2CC")
    for tarea, d in por_tarea.items():
        m = d[(d["B"] != "nan") & (d["B"] != "")]
        if m.empty:
            continue
        cats = sorted(set(m["A"]) | set(m["B"]))
        ct = pd.crosstab(m["A"], m["B"]).reindex(index=cats, columns=cats, fill_value=0)
        ws = wb.create_sheet(f"kappa_{tarea}"[:31])
        ws["A1"] = f"Acuerdo interanotador - {tarea}"
        ws["A1"].font = Font(bold=True, size=13)
        ws["A3"] = "Matriz A (filas) x B (columnas). Los totales y el kappa son formulas: "
        ws["A4"] = "si cambias un numero de la matriz, el kappa se recalcula solo."
        f0 = 6                       # fila del encabezado de la matriz
        ws.cell(f0, 1, "A \\ B").font = neg
        for j, c in enumerate(cats):
            x = ws.cell(f0, 2 + j, c); x.font = Font(bold=True, color="FFFFFF"); x.fill = cab
        for i, c in enumerate(cats):
            x = ws.cell(f0 + 1 + i, 1, c); x.font = Font(bold=True, color="FFFFFF"); x.fill = cab
            for j, c2 in enumerate(cats):
                ws.cell(f0 + 1 + i, 2 + j, int(ct.loc[c, c2]))
        k = len(cats)
        c_ini, c_fin = get_column_letter(2), get_column_letter(1 + k)
        f_ini, f_fin = f0 + 1, f0 + k
        ftot = f_fin + 1
        ws.cell(ftot, 1, "total B").font = neg
        for j in range(k):
            col = get_column_letter(2 + j)
            ws.cell(ftot, 2 + j, f"=SUM({col}{f_ini}:{col}{f_fin})")
        ctot = get_column_letter(2 + k)
        ws.cell(f0, 2 + k, "total A").font = neg
        for i in range(k):
            ws.cell(f_ini + i, 2 + k, f"=SUM({c_ini}{f_ini + i}:{c_fin}{f_ini + i})")
        ws.cell(ftot, 2 + k, f"=SUM({c_ini}{f_ini}:{c_fin}{f_fin})")
        base = ftot + 2
        celdas = {}
        ws.cell(base, 1, "n total").font = neg
        ws.cell(base, 2, f"={ctot}{ftot}"); celdas["n"] = f"B{base}"
        ws.cell(base + 1, 1, "aciertos (diagonal)").font = neg
        diag = "+".join(f"{get_column_letter(2 + i)}{f_ini + i}" for i in range(k))
        ws.cell(base + 1, 2, f"={diag}")
        ws.cell(base + 2, 1, "Po  acuerdo observado").font = neg
        ws.cell(base + 2, 2, f"=B{base + 1}/B{base}")
        ws.cell(base + 3, 1, "Pe  acuerdo esperado por azar").font = neg
        prod = "+".join(f"({get_column_letter(2 + i)}{ftot}/B{base})*({ctot}{f_ini + i}/B{base})"
                        for i in range(k))
        ws.cell(base + 3, 2, f"={prod}")
        c = ws.cell(base + 4, 1, "KAPPA DE COHEN"); c.font = Font(bold=True, size=12)
        c2 = ws.cell(base + 4, 2, f"=(B{base + 2}-B{base + 3})/(1-B{base + 3})")
        c2.font = Font(bold=True, size=12); c2.fill = am
        c2.number_format = "0.000"
        ws.cell(base + 2, 2).number_format = "0.0000"
        ws.cell(base + 3, 2).number_format = "0.0000"
        ws.cell(base + 6, 1, "Formula:  kappa = (Po - Pe) / (1 - Pe)")
        ws.cell(base + 7, 1, "Po = suma de la diagonal / n")
        ws.cell(base + 8, 1, "Pe = suma sobre categorias de (total de esa fila/n) x (total de esa columna/n)")
        ws.column_dimensions["A"].width = 34
        for j in range(k + 1):
            ws.column_dimensions[get_column_letter(2 + j)].width = 15
        for row in ws.iter_rows(min_row=f0, max_row=ftot):
            for cc in row:
                cc.alignment = Alignment(horizontal="center")
        ws["A3"].alignment = Alignment(horizontal="left")
    if wb.sheetnames:
        wb.save(ruta)
        log.info("Libro auditable: %s", ruta)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    ap = comun.parser_base("Etapa 09: evaluacion de la anotacion humana.")
    ap.add_argument("--tarea", choices=list(TAREAS), help="evaluar solo una tarea")
    ap.add_argument("--sin-adjudicacion", action="store_true",
                    help="ignorar ADJUDICAR_desacuerdos.xlsx")
    args = ap.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if not DIR_ANOTADOS.exists():
        log.error("No existe %s. Nada que evaluar.", DIR_ANOTADOS)
        return

    adj = {} if args.sin_adjudicacion else cargar_adjudicacion(log)
    tareas = [args.tarea] if args.tarea else list(TAREAS)

    resumen, filas_acuerdo, precisiones, por_tarea = {}, [], {}, {}

    for tarea in tareas:
        d = cargar_tarea(tarea, log)
        if d is None:
            continue
        d = construir_referencia(d, tarea, adj, log)
        por_tarea[tarea] = d

        ac = acuerdo_interanotador(d)
        ac["tarea"] = tarea
        filas_acuerdo.append(ac)
        log.info("%s | n=%s | acuerdo bruto %.1f%% | kappa %.3f (%s)",
                 tarea, ac["n"], 100 * (ac["acuerdo_bruto"] or 0),
                 ac["kappa"] if ac["kappa"] is not None else float("nan"),
                 ac["interpretacion"])

        pr = precision_por_clase(d, tarea)
        precisiones[tarea] = pr
        comun.guardar_tabla(pr, f"s09_precision_{tarea}", log)
        mc = matriz_confusion(d)
        comun.guardar_tabla(mc.reset_index(), f"s09_matriz_{tarea}", log)

        n_gold = int(d["gold"].notna().sum())
        resumen[tarea] = {
            "n_muestra": int(len(d)),
            "n_en_estandar_de_referencia": n_gold,
            "n_desacuerdos_sin_adjudicar": int((d["origen_gold"] == "desacuerdo_sin_adjudicar").sum()),
            "desglose_origen": d["origen_gold"].value_counts().to_dict(),
            "acuerdo_interanotador": {k: v for k, v in ac.items() if k != "tarea"},
            "precision_por_clase": pr.to_dict(orient="records"),
        }

    # --- reponderacion de polaridad ---
    if "polaridad" in por_tarea:
        pesos = PESOS_POLARIDAD_RESPALDO.copy()
        rp = cfg.DIR_TABLAS / "s05_distribucion_polaridad.csv"
        if rp.exists():
            dd = pd.read_csv(rp)
            pesos = dict(zip(dd["polaridad"], dd["n"]))
        rep = reponderar_polaridad(por_tarea["polaridad"], pesos)
        comun.guardar_tabla(rep, "s09_polaridad_corregida", log)
        resumen.setdefault("polaridad", {})["exactitud_reponderada"] = rep.attrs["exactitud_reponderada"]
        resumen["polaridad"]["prevalencia_corregida"] = rep.to_dict(orient="records")
        log.info("Polaridad: exactitud reponderada al corpus %.3f",
                 rep.attrs["exactitud_reponderada"])
        for _, r in rep.iterrows():
            log.info("   %-9s modelo %5.1f%% -> corregido %5.1f%%",
                     r["clase"], r["pct_modelo"], r["pct_corregido"])

    # --- precision de mencion ---
    if "entidad" in precisiones:
        pm = precision_mencion(precisiones["entidad"], log)
        if not pm.empty:
            comun.guardar_tabla(pm, "s09_precision_mencion_por_categoria", log)
            resumen["precision_mencion"] = {
                "umbral_analizable": UMBRAL_PRECISION_ANALIZABLE,
                "categorias": pm.to_dict(orient="records"),
                "analizables": pm.loc[pm["analizable"], "categoria"].tolist(),
                "excluidas": pm.loc[~pm["analizable"], "categoria"].tolist(),
            }
            log.info("Categorias analizables (precision de mencion >= %.2f): %s",
                     UMBRAL_PRECISION_ANALIZABLE,
                     ", ".join(pm.loc[pm["analizable"], "categoria"]))

    if filas_acuerdo:
        comun.guardar_tabla(pd.DataFrame(filas_acuerdo), "s09_acuerdo_interanotador", log)
    exportar_auditable(por_tarea, cfg.DIR_VALIDACION / "s09_calculo_auditable.xlsx", log)
    comun.guardar_metricas(resumen, "s09_validacion_humana", log)
    log.info("ETAPA 09 COMPLETADA.")


if __name__ == "__main__":
    main()
