# -*- coding: utf-8 -*-
"""
ETAPA 04 — Clasificacion de entidades: cuentas y menciones a categorias iCARE.

Que resuelve del informe del asesor
-----------------------------------
* Correccion #8 / seccion 2.3.6: "Definir reglas de mapeo de cuentas y
  menciones a las categorias iCARE; explicar clasificacion manual vs.
  automatica, evaluacion interanotador si hubo anotacion humana y
  precision de la clasificacion."

Dos niveles de entidad, y por que importa la diferencia
------------------------------------------------------
  entidad_autor     categoria de QUIEN ESCRIBE el tuit.
  entidad_mencionada categoria de QUIEN ES CITADO en el tuit.

  La version 1 del pipeline cruzaba polaridad contra `entidad_autor` y
  presentaba el resultado como "polaridad hacia la entidad". Eso es
  incorrecto: mide el tono de los mensajes ESCRITOS POR esas cuentas, no
  el tono de los mensajes DIRIGIDOS A ellas. Como el manuscrito habla de
  "iCARE entities cited in Twitter/X", la unidad correcta es la MENCION.

  Este script produce los dos y deja claro cual alimenta cada tabla.
  La Tabla 3 del manuscrito pasa a construirse sobre menciones. Las
  cifras cambiaran respecto de la version anterior del paper: es una
  correccion, no una discrepancia.

Cascada de resolucion (de mas fiable a menos)
---------------------------------------------
  R1. Diccionario curado de cuentas conocidas
      (recursos/cuentas_conocidas.csv). Determinista y auditable: cada
      fila lleva su justificacion. Es el componente "manual".
  R2. Reglas lexicas de alta precision sobre la descripcion del perfil
      (p. ej. variantes de OMS/WHO/OPS). Heredadas de la version 1.
  R3. Clasificacion automatica zero-shot de la descripcion del perfil con
      joeddav/xlm-roberta-large-xnli, con umbral de confianza y margen
      minimo respecto de la segunda categoria. Es el componente
      "automatico".
  R4. Sin resolver -> sin_categoria. NO se fuerza una asignacion.

  El script reporta que porcentaje resuelve cada regla: esa es la
  respuesta literal a "proporcion manual vs. automatica".

Coste
-----
El zero-shot se aplica UNA VEZ por descripcion de perfil DISTINTA, no una
vez por tuit. En un corpus donde los mismos autores tuitean muchas veces
esto reduce el computo en uno o dos ordenes de magnitud respecto de la
version 1, que clasificaba fila por fila.

VALIDACION PENDIENTE. No hay anotacion humana de las categorias. La etapa
09 genera la muestra lista para anotar; mientras no se devuelva anotada,
el manuscrito debe declarar que la precision de la clasificacion de
entidades no ha sido verificada, y no puede reportar acuerdo
interanotador.

Salida
------
  intermedios/s04_entidades.parquet   (una fila por tuit)
  intermedios/s04_menciones.parquet   (una fila por MENCION: unidad de analisis)
  resultados/tablas/s04_resolucion_entidades.csv
  resultados/tablas/s04_entidades_por_autor.csv
  resultados/tablas/s04_top_cuentas_por_categoria.csv
  resultados/metricas/s04_entidades.json
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
import comun

ETAPA = "s04"

# --- R3: hipotesis para el clasificador zero-shot -------------------------
# Frases descriptivas en espanol, una por categoria. El modelo XNLI evalua
# "esta descripcion de perfil trata de {hipotesis}". Frases largas y
# especificas rinden mejor que etiquetas de una palabra.
HIPOTESIS = {
    "lider_nacional": "un presidente, ministro o cargo del gobierno nacional",
    "lider_regional": "un alcalde, gobernador o autoridad de una ciudad o departamento",
    "autoridad_salud": "un ministerio de salud, instituto nacional de salud u organismo sanitario oficial",
    "profesional_salud": "un medico, enfermera u otro profesional de la salud",
    "oms": "la Organizacion Mundial de la Salud",
    "ciencia": "un cientifico, investigador, universidad o revista cientifica",
    "famosos": "un actor, cantante, musico o deportista famoso",
    "lider_comunitario": "un lider religioso, comunitario, sindical o cultural",
    "medios": "un periodista, periodico, canal de television, emisora o medio de comunicacion",
}

# --- R2: reglas lexicas de alta precision ---------------------------------
REGLAS_LEXICAS = {
    "oms": [r"\boms\b", r"\bo\.m\.s\.?\b", r"\bwho\b", r"\bw\.h\.o\.?\b",
            r"\bops\b", r"\bo\.p\.s\.?\b",
            r"organizaci[oó]n mundial de la salud",
            r"pan american health organization"],
}


def compilar_reglas():
    return {cat: [re.compile(p, re.IGNORECASE) for p in pats]
            for cat, pats in REGLAS_LEXICAS.items()}


def aplicar_reglas_lexicas(texto: str, reglas) -> str | None:
    if not isinstance(texto, str) or not texto.strip():
        return None
    for cat, pats in reglas.items():
        if any(p.search(texto) for p in pats):
            return cat
    return None


def clasificar_zero_shot(descripciones: list[str], log) -> list[tuple[str, float, float]]:
    """Una llamada por descripcion con todas las hipotesis a la vez.

    Devuelve (categoria, score, margen). Se exige score >= umbral Y
    margen >= margen_min sobre la segunda categoria; si no, sin_categoria.
    El margen evita asignaciones en las que el modelo esta esencialmente
    indeciso entre dos categorias.
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    log.info("Zero-shot con %s en %s", cfg.MODELO_ENTIDADES,
             "GPU" if device == 0 else "CPU (sera muy lento)")
    # [LOCAL] fp16 para que xlm-roberta-large quepa comodo en 6 GB.
    kw = {}
    if device == 0 and cfg.USAR_FP16:
        kw["torch_dtype"] = torch.float16
    clf = pipeline("zero-shot-classification", model=cfg.MODELO_ENTIDADES,
                   device=device, **kw)
    if device == 0:
        libre, total = torch.cuda.mem_get_info()
        log.info("VRAM: %.1f GB libres de %.1f GB. Lote %d, que son %d secuencias "
                 "por paso (lote x %d hipotesis).",
                 libre / 1e9, total / 1e9, cfg.ENTIDADES_BATCH,
                 cfg.ENTIDADES_BATCH * len(HIPOTESIS), len(HIPOTESIS))
        log.info("Si aparece 'CUDA out of memory', bajar ENTIDADES_BATCH en config.py.")
        log.info("Si va demasiado lento, config.MODELO_ENTIDADES_ALTERNATIVO es la "
                 "mitad de grande; declarar en Methods el que se acabe usando.")

    claves = list(HIPOTESIS.keys())
    etiquetas = [HIPOTESIS[k] for k in claves]
    inv = {v: k for k, v in HIPOTESIS.items()}

    out = []
    lote = cfg.ENTIDADES_BATCH
    for i in range(0, len(descripciones), lote):
        trozo = descripciones[i:i + lote]
        res = clf(trozo, etiquetas, multi_label=False,
                  hypothesis_template="Esta descripcion de perfil corresponde a {}.")
        if isinstance(res, dict):
            res = [res]
        for r in res:
            s1, s2 = r["scores"][0], (r["scores"][1] if len(r["scores"]) > 1 else 0.0)
            cat = inv[r["labels"][0]]
            out.append((cat, float(s1), float(s1 - s2)))
        if (i // lote) % 50 == 0:
            log.info("  zero-shot %s/%s descripciones", f"{i + len(trozo):,}",
                     f"{len(descripciones):,}")
    return out


def tablas_derivadas_de_menciones(menciones: pd.DataFrame, log) -> pd.DataFrame:
    """Tablas que se derivan de la tabla de menciones validas.

    Se generan AQUI, dentro del pipeline, y nunca a mano: la etapa 08
    (figuras) y la etapa 09 (validacion) las consumen, y el manuscrito exige
    que toda cifra publicada salga de un script reproducible.

      * s04_resolucion_menciones_por_categoria.csv
            por categoria, cuantas menciones resolvio el diccionario curado y
            cuantas el zero-shot. Es lo que permite estimar en la etapa 09 la
            precision de la categoria a nivel de MENCION.
      * s04_top_menciones_corpus.csv
            las cuentas mas mencionadas del corpus colombiano, con su
            categoria y la regla que la resolvio.
    """
    tab = (pd.crosstab(menciones["categoria_mencionada"], menciones["origen_mencion"])
           .reset_index().rename(columns={"categoria_mencionada": "categoria"}))
    for col in ("R1_diccionario_curado", "R3_autor_del_corpus"):
        if col not in tab.columns:
            tab[col] = 0
    tab["n_menciones"] = tab["R1_diccionario_curado"] + tab["R3_autor_del_corpus"]
    tab["pct_curado"] = (100 * tab["R1_diccionario_curado"] / tab["n_menciones"]).round(2)
    tab = tab.sort_values("n_menciones", ascending=False)
    comun.guardar_tabla(tab, "s04_resolucion_menciones_por_categoria", log)
    log.info("Resolucion de menciones por categoria:\n%s", tab.to_string(index=False))

    total = len(menciones)
    top = (menciones.groupby("handle_mencionado")
           .agg(n_menciones=("handle_mencionado", "size"),
                categoria=("categoria_mencionada", "first"),
                origen=("origen_mencion", "first"))
           .reset_index().rename(columns={"handle_mencionado": "handle"})
           .sort_values("n_menciones", ascending=False)
           .head(cfg.TOP_N_MENCIONES))
    top["pct_del_total"] = (100 * top["n_menciones"] / total).round(2)
    top = top[["handle", "n_menciones", "pct_del_total", "categoria", "origen"]]
    comun.guardar_tabla(top, "s04_top_menciones_corpus", log)
    log.info("Cuentas mas mencionadas (top %s, %.1f%% de las menciones validas):\n%s",
             cfg.TOP_N_MENCIONES, top["pct_del_total"].sum(), top.to_string(index=False))
    return tab


def main() -> None:
    p = comun.parser_base(__doc__)
    p.add_argument("--sin-zeroshot", action="store_true",
                   help="Solo diccionario y reglas lexicas. Para diagnostico rapido.")
    p.add_argument("--solo-tablas", action="store_true",
                   help="Regenerar solo las tablas derivadas a partir de "
                        "s04_menciones.parquet, sin volver a correr el zero-shot.")
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if args.solo_tablas:
        if not cfg.PQ_MENCIONES.exists():
            log.error("Falta %s: no hay de donde derivar las tablas.", cfg.PQ_MENCIONES.name)
            sys.exit(1)
        log.info("Modo --solo-tablas: se reconstruyen las tablas derivadas sin "
                 "recalcular la clasificacion.")
        tablas_derivadas_de_menciones(pd.read_parquet(cfg.PQ_MENCIONES), log)
        log.info("ETAPA 04 (solo tablas) COMPLETADA.")
        return

    if comun.debe_saltar(cfg.PQ_ENTIDADES, args.forzar, log):
        return
    if not cfg.PQ_COLOMBIA.exists():
        log.error("Falta %s. Ejecutar antes la etapa 03d.", cfg.PQ_COLOMBIA.name)
        sys.exit(1)

    df = pd.read_parquet(cfg.PQ_COLOMBIA)
    log.info("Corpus colombiano: %s tuits", f"{len(df):,}")

    # ---------------- R1: diccionario curado ----------------
    dic = pd.read_csv(cfg.ARCHIVO_CUENTAS_CONOCIDAS)
    dic["handle"] = dic["handle"].str.strip().str.lower()
    mapa_dic = dict(zip(dic["handle"], dic["categoria"]))
    desconocidas = set(dic["categoria"]) - set(cfg.CATEGORIAS)
    if desconocidas:
        log.error("cuentas_conocidas.csv usa categorias que no estan en config: %s", desconocidas)
        sys.exit(1)
    log.info("Diccionario curado: %d cuentas en %d categorias",
             len(mapa_dic), dic["categoria"].nunique())

    reglas = compilar_reglas()
    df["autor_n"] = df["autor"].fillna("").str.strip().str.lower()

    # ---------------- catalogo de autores distintos ----------------
    autores = (df.groupby("autor_n")
                 .agg(descripcion=("autor_descripcion", "first"),
                      nombre=("autor_nombre", "first"),
                      n_tuits=("texto", "size"),
                      seguidores=("autor_seguidores", "max"))
                 .reset_index())
    autores = autores[autores["autor_n"] != ""]
    log.info("Autores distintos en el corpus colombiano: %s", f"{len(autores):,}")

    # Las columnas de resultado se crean como 'object' a proposito: se van
    # rellenando con cadenas por tramos y una columna float64 rechazaria la
    # asignacion.
    autores["categoria"] = pd.Series([None] * len(autores), dtype="object")
    autores["origen"] = pd.Series([None] * len(autores), dtype="object")
    autores["score"] = np.nan
    autores["margen"] = np.nan

    # R1
    m1 = autores["autor_n"].map(mapa_dic)
    hit1 = m1.notna()
    autores.loc[hit1, "categoria"] = m1[hit1].values
    autores.loc[hit1, "origen"] = "R1_diccionario_curado"
    autores.loc[hit1, "score"] = 1.0
    autores.loc[hit1, "margen"] = 1.0
    log.info("R1 diccionario curado: %s autores resueltos", f"{int(hit1.sum()):,}")

    # R2
    pend = autores["categoria"].isna()
    m2 = autores.loc[pend, "descripcion"].map(lambda t: aplicar_reglas_lexicas(t, reglas))
    hit2 = m2.notna()
    idx2 = m2[hit2].index
    autores.loc[idx2, "categoria"] = m2[hit2].values
    autores.loc[idx2, "origen"] = "R2_reglas_lexicas"
    autores.loc[idx2, "score"] = 1.0
    autores.loc[idx2, "margen"] = 1.0
    log.info("R2 reglas lexicas: %s autores resueltos", f"{len(idx2):,}")

    # R3
    pend = autores["categoria"].isna()
    sin_desc = pend & (autores["descripcion"].isna()
                       | (autores["descripcion"].astype(str).str.strip() == ""))
    autores.loc[sin_desc, "categoria"] = cfg.SIN_DESCRIPCION
    autores.loc[sin_desc, "origen"] = "R4_sin_descripcion"
    log.info("Autores sin descripcion de perfil (no clasificables): %s",
             f"{int(sin_desc.sum()):,}")

    pend = autores["categoria"].isna()
    n_pend = int(pend.sum())
    log.info("Pendientes para zero-shot: %s descripciones distintas", f"{n_pend:,}")

    if args.sin_zeroshot:
        log.warning("--sin-zeroshot activo: %s autores quedan sin categoria. "
                    "NO usar para resultados definitivos.", f"{n_pend:,}")
        autores.loc[pend, "categoria"] = cfg.SIN_CATEGORIA
        autores.loc[pend, "origen"] = "R4_no_ejecutado"
    elif n_pend:
        with comun.Cronometro(log, f"zero-shot sobre {n_pend:,} descripciones"):
            res = clasificar_zero_shot(
                autores.loc[pend, "descripcion"].astype(str).tolist(), log)
        cats = np.array([r[0] for r in res])
        scores = np.array([r[1] for r in res])
        margenes = np.array([r[2] for r in res])
        acepta = (scores >= cfg.ENTIDADES_UMBRAL_MIN) & (margenes >= cfg.ENTIDADES_MARGEN_MIN)
        cats_final = np.where(acepta, cats, cfg.SIN_CATEGORIA)
        idx = autores.index[pend]
        autores.loc[idx, "categoria"] = cats_final
        autores.loc[idx, "origen"] = np.where(acepta, "R3_zero_shot", "R4_baja_confianza")
        autores.loc[idx, "score"] = scores
        autores.loc[idx, "margen"] = margenes
        log.info("Zero-shot: %s aceptados, %s rechazados por umbral %.2f / margen %.2f",
                 f"{int(acepta.sum()):,}", f"{int((~acepta).sum()):,}",
                 cfg.ENTIDADES_UMBRAL_MIN, cfg.ENTIDADES_MARGEN_MIN)

    # ---------------- proporcion manual vs automatica ----------------
    res_aut = (autores["origen"].value_counts(dropna=False)
               .rename_axis("regla").reset_index(name="n_autores"))
    res_aut["pct"] = (res_aut["n_autores"] / len(autores) * 100).round(2)
    comun.guardar_tabla(res_aut, "s04_resolucion_entidades", log)
    log.info("Resolucion de entidades por regla:\n%s", res_aut.to_string(index=False))

    dist = (autores["categoria"].value_counts()
            .rename_axis("categoria").reset_index(name="n_autores"))
    dist["etiqueta_en"] = dist["categoria"].map(cfg.ETIQUETA_EN).fillna(dist["categoria"])
    comun.guardar_tabla(dist, "s04_entidades_por_autor", log)
    log.info("Distribucion de categorias de autor:\n%s", dist.to_string(index=False))

    top = (autores[autores["categoria"].isin(cfg.CATEGORIAS_TWITTER)]
           .sort_values(["categoria", "n_tuits"], ascending=[True, False])
           .groupby("categoria").head(15)
           [["categoria", "autor_n", "nombre", "n_tuits", "seguidores", "origen", "score"]])
    comun.guardar_tabla(top, "s04_top_cuentas_por_categoria", log)
    log.info("Guardado s04_top_cuentas_por_categoria.csv — REVISARLO A MANO: es la "
             "forma barata de detectar errores sistematicos del zero-shot.")

    # ---------------- entidad del autor, por tuit ----------------
    mapa_autor_cat = dict(zip(autores["autor_n"], autores["categoria"]))
    df["entidad_autor"] = df["autor_n"].map(mapa_autor_cat).fillna(cfg.SIN_CATEGORIA)

    # ---------------- entidad mencionada: tabla de menciones ----------------
    # Un handle mencionado se resuelve por el diccionario curado y, si no,
    # por la categoria del mismo handle como AUTOR del corpus (si aparece).
    mapa_mencion = dict(mapa_autor_cat)
    mapa_mencion.update(mapa_dic)   # el diccionario curado tiene prioridad

    with comun.Cronometro(log, "expandiendo menciones a una fila por mencion"):
        expl = df[["autor_n", "fecha", "menciones"]].copy()
        expl["fila_tuit"] = np.arange(len(expl))
        expl = expl.explode("menciones").dropna(subset=["menciones"])
        expl = expl.rename(columns={"menciones": "handle_mencionado"})
        expl["handle_mencionado"] = expl["handle_mencionado"].str.lower()
        expl["categoria_mencionada"] = expl["handle_mencionado"].map(mapa_mencion)
        # Un handle puede estar en el mapa pero con categoria no sustantiva
        # (sin_categoria / sin_descripcion): eso NO es haberlo resuelto.
        sustantiva = expl["categoria_mencionada"].isin(cfg.CATEGORIAS_TWITTER)
        expl["origen_mencion"] = np.where(
            expl["handle_mencionado"].isin(mapa_dic) & sustantiva, "R1_diccionario_curado",
            np.where(sustantiva, "R3_autor_del_corpus", "sin_resolver"))
        expl["categoria_mencionada"] = expl["categoria_mencionada"].fillna(cfg.SIN_CATEGORIA)

    n_men = len(expl)
    resueltas = expl["categoria_mencionada"].isin(cfg.CATEGORIAS_TWITTER)
    log.info("Menciones totales: %s | resueltas a una categoria iCARE: %s (%.1f%%)",
             f"{n_men:,}", f"{int(resueltas.sum()):,}", 100 * resueltas.mean())

    res_men = (expl["origen_mencion"].value_counts()
               .rename_axis("regla").reset_index(name="n_menciones"))
    res_men["pct"] = (res_men["n_menciones"] / n_men * 100).round(2)
    comun.guardar_tabla(res_men, "s04_resolucion_menciones", log)
    log.info("Resolucion de menciones:\n%s", res_men.to_string(index=False))

    # menciones validas = las que caen en una categoria comparable con iCARE
    menciones_validas = expl[resueltas].copy()
    comun.guardar_parquet(menciones_validas, cfg.PQ_MENCIONES, log)

    tablas_derivadas_de_menciones(menciones_validas, log)

    # ---------------- cuantas categorias distintas menciona cada tuit ----------------
    # Sirve para el analisis de sensibilidad de la etapa 07: en los tuits
    # que mencionan UNA sola categoria, atribuir la polaridad del tuit a
    # esa categoria es mucho menos discutible.
    por_tuit = (menciones_validas.groupby("fila_tuit")["categoria_mencionada"]
                .agg(["nunique", lambda s: sorted(set(s))]))
    por_tuit.columns = ["n_categorias_mencionadas", "categorias_mencionadas"]
    df["n_categorias_mencionadas"] = (
        por_tuit["n_categorias_mencionadas"].reindex(np.arange(len(df))).fillna(0).astype(int).values)
    df["categorias_mencionadas"] = (
        por_tuit["categorias_mencionadas"].reindex(np.arange(len(df))).apply(
            lambda x: x if isinstance(x, list) else []).values)
    n_una = int((df["n_categorias_mencionadas"] == 1).sum())
    log.info("Tuits que mencionan exactamente UNA categoria iCARE: %s (%.1f%% del corpus "
             "colombiano) — es la submuestra del analisis de sensibilidad",
             f"{n_una:,}", 100 * n_una / len(df))

    df = df.drop(columns=["autor_n"])
    comun.guardar_parquet(df, cfg.PQ_ENTIDADES, log)

    comun.registrar_funnel(ETAPA, "E11_con_entidad", int((df["n_categorias_mencionadas"] > 0).sum()),
                           "Tuits que citan al menos una categoria iCARE identificada")
    comun.registrar_funnel(ETAPA, "E12_menciones_validas", int(len(menciones_validas)),
                           "Menciones validas (unidad de analisis de la Tabla 3)")

    comun.guardar_metricas({
        "n_autores_distintos": int(len(autores)),
        "resolucion_autores": res_aut.to_dict("records"),
        "distribucion_categorias_autor": dist.to_dict("records"),
        "n_menciones_totales": n_men,
        "n_menciones_validas": int(resueltas.sum()),
        "pct_menciones_resueltas": round(100 * resueltas.mean(), 2),
        "resolucion_menciones": res_men.to_dict("records"),
        "n_tuits_con_una_sola_categoria": n_una,
        "modelo_zero_shot": cfg.MODELO_ENTIDADES,
        "umbral_score": cfg.ENTIDADES_UMBRAL_MIN,
        "margen_minimo": cfg.ENTIDADES_MARGEN_MIN,
        "hipotesis": HIPOTESIS,
        "validacion_humana": "PENDIENTE — ver etapa 09. Sin ella no se puede reportar "
                             "precision de la clasificacion ni acuerdo interanotador.",
    }, "s04_entidades", log)

    log.info("ETAPA 04 COMPLETADA.")


if __name__ == "__main__":
    main()
