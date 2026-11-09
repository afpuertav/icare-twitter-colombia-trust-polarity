# -*- coding: utf-8 -*-
"""
ETAPA 07b — Consolidacion del embudo, muestras de validacion y auditoria final.

Se ejecuta DESPUES de la etapa 07 y ANTES de la 08 (figuras), porque la
Figura 1 y la Figura 2 leen el embudo consolidado que produce este script.

Tres cosas
----------
1. EMBUDO CONSOLIDADO
   Ordena todas las filas que las etapas fueron registrando y calcula el
   porcentaje retenido respecto del inicio y respecto del paso anterior.
   Es la tabla que responde a la correccion #9 y la que alimenta la
   Figura 2.

2. MUESTRAS DE VALIDACION MANUAL
   El asesor exige metricas de la clasificacion (secciones 2.3.5, 2.3.6 y
   2.3.7) y acuerdo interanotador si hubo anotacion humana. No hay
   anotacion. Este script deja las muestras aleatorias estratificadas
   listas para anotar, en Excel, con instrucciones y con la columna de
   respuesta vacia:

     validacion/muestra_polaridad.xlsx   400 tuits
     validacion/muestra_entidad.xlsx     400 perfiles
     validacion/muestra_origen_geo.xlsx  400 tuits

   Al devolverlas anotadas, `s09_evaluar_anotacion.py` calcula precision,
   recall, F1 y matriz de confusion por tarea, y kappa de Cohen si hay dos
   anotadores.

   MIENTRAS TANTO el manuscrito debe declarar en Limitations que la
   clasificacion automatica (origen geografico, entidad y polaridad) no
   ha sido validada contra anotacion humana.

3. AUDITORIA DE CIERRE
   Comprueba que existan todos los insumos que el manuscrito necesita y
   lista lo que falta, para no descubrirlo al escribir.

Salida
------
  resultados/tablas/s09_funnel_completo.csv
  resultados/validacion/muestra_*.xlsx
  resultados/tablas/s09_auditoria_cierre.csv
  resultados/metricas/s09_consolidacion.json
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

ETAPA = "s07b"

# Orden canonico del embudo. Si una etapa no se ejecuto, su fila
# simplemente no aparece.
# 'secuencial' = filtro que se aplica sobre el resultado del paso anterior;
# solo en esos tiene sentido el "% del paso anterior". Las evidencias
# paralelas y los cambios de unidad de analisis se marcan aparte para que
# la Figura 2 no sugiera una cadena que no existe.
ORDEN_FUNNEL = [
    ("s01", "E00_declarado"),
    ("s01", "E01_parseado"),
    ("s01", "E02_con_fecha"),
    ("s01", "E03_sin_retweets"),
    ("s01", "E04_deduplicado"),
    ("s01", "E05_espanol"),
    ("s01", "E06_ventana"),
    ("s02", "E07_preprocesado"),
    ("s03a", "E08_seed_gazetteer"),
    ("s03c", "E09_seed_grafo"),
    ("s03d", "E10_corpus_colombia"),
    ("s04", "E11_con_entidad"),
    ("s04", "E12_menciones_validas"),
    ("s05", "E13_corpus_analitico"),
]

def muestra_estratificada(df: pd.DataFrame, col: str, n_total: int,
                          semilla: int) -> pd.DataFrame:
    """Muestreo estratificado por `col`, repartiendo n_total entre los
    estratos y barajando el resultado.

    Se implementa con indices en vez de groupby().apply() a proposito:
    desde pandas 2.2 apply() excluye la columna de agrupacion del
    DataFrame que recibe la funcion, lo que hacia desaparecer la columna
    justo despues de muestrear.
    """
    grupos = [g for _, g in df.groupby(col, sort=True)]
    if not grupos:
        return df.head(0)
    por_grupo = max(1, n_total // len(grupos))
    partes = [g.sample(min(len(g), por_grupo), random_state=semilla) for g in grupos]
    out = pd.concat(partes)
    # se baraja para que no se anote por bloques de una misma clase
    return out.sample(frac=1, random_state=semilla)


INSTRUCCIONES = {
    "polaridad": [
        "TAREA: leer el tuit y clasificar el TONO GENERAL del mensaje completo.",
        "",
        "Escribir en la columna 'anotacion' exactamente una de:",
        "   Negative   el tuit expresa critica, enfado, miedo, queja o rechazo",
        "   Neutral    informativo o descriptivo, sin carga evaluativa",
        "   Positive   apoyo, agradecimiento, esperanza, elogio",
        "",
        "Importante: se clasifica el TONO DEL TUIT COMPLETO, no la opinion",
        "sobre una cuenta concreta que aparezca mencionada.",
        "",
        "No mirar la columna 'prediccion_modelo' antes de decidir. Esta ahi",
        "para el calculo posterior de metricas, no como ayuda.",
        "",
        "Si el tuit es ilegible o esta en otro idioma, escribir: descartar",
    ],
    "entidad": [
        "TAREA: leer la descripcion del perfil y decidir a que categoria de",
        "fuente de informacion corresponde la cuenta.",
        "",
        "Escribir en 'anotacion' una de las claves de la hoja 'categorias'.",
        "Si no corresponde a ninguna, o la descripcion no alcanza para",
        "decidirlo, escribir: sin_categoria",
        "",
        "Criterio: que ES la cuenta, no de que habla. Un ciudadano que opina",
        "sobre politica no es 'lider_nacional'.",
        "",
        "No mirar 'prediccion_modelo' antes de decidir.",
    ],
    "origen_geo": [
        "TAREA: decidir si el tuit fue probablemente escrito por alguien en",
        "Colombia, usando el texto, la descripcion del perfil y el nombre.",
        "",
        "Escribir en 'anotacion': si | no | no_se",
        "",
        "Usar 'no_se' sin reparos: es informacion valiosa y evita forzar",
        "juicios que inflarian artificialmente las metricas.",
        "",
        "No mirar 'prediccion_modelo' antes de decidir.",
    ],
}


def escribir_muestra(df: pd.DataFrame, nombre: str, tarea: str, log,
                     extra: pd.DataFrame | None = None) -> None:
    ruta = cfg.DIR_VALIDACION / f"muestra_{nombre}.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        pd.DataFrame({"INSTRUCCIONES": INSTRUCCIONES[tarea]}).to_excel(
            w, sheet_name="instrucciones", index=False)
        df.to_excel(w, sheet_name="muestra", index=False)
        if extra is not None:
            extra.to_excel(w, sheet_name="categorias", index=False)
    log.info("Muestra de validacion: %s (%d filas)", ruta.name, len(df))
    if cfg.VALIDACION_DOBLE_ANOTADOR:
        ruta2 = cfg.DIR_VALIDACION / f"muestra_{nombre}_anotador2.xlsx"
        import shutil
        shutil.copy(ruta, ruta2)
        log.info("Copia para el segundo anotador: %s", ruta2.name)


def main() -> None:
    p = comun.parser_base(__doc__)
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()
    rng = np.random.default_rng(cfg.SEMILLA)

    # ---------------- 1. embudo ----------------
    fun = comun.construir_funnel_ordenado(ORDEN_FUNNEL)
    if fun.empty:
        log.warning("El embudo esta vacio: no se ha ejecutado ninguna etapa aun.")
    else:
        comun.guardar_tabla(fun, "s09_funnel_completo", log)
        log.info("\nEMBUDO DEL CORPUS (correccion #9):\n%s", fun.to_string(index=False))

    ica = cfg.DIR_TABLAS / "s06_flujo_icare.csv"
    if ica.exists():
        log.info("\nFLUJO iCARE:\n%s", pd.read_csv(ica).to_string(index=False))

    # ---------------- 2. muestras de validacion ----------------
    if cfg.PQ_ANALITICO.exists():
        tw = pd.read_parquet(cfg.PQ_ANALITICO)

        # polaridad: estratificada por clase predicha
        m = muestra_estratificada(tw, "polaridad", cfg.VALIDACION_N_POLARIDAD, cfg.SEMILLA)
        escribir_muestra(pd.DataFrame({
            "id_muestra": np.arange(1, len(m) + 1),
            "texto": m["texto"].values,
            "fecha": m["fecha"].values,
            "anotacion": "",
            "prediccion_modelo": m["polaridad"].values,
            "confianza_modelo": m["confianza_polaridad"].round(3).values,
        }), "polaridad", "polaridad", log)

        # origen geografico
        k = min(cfg.VALIDACION_N_GEO, len(tw))
        g = tw.sample(k, random_state=cfg.SEMILLA + 1)
        escribir_muestra(pd.DataFrame({
            "id_muestra": np.arange(1, len(g) + 1),
            "texto": g["texto"].values,
            "autor": g["autor"].values,
            "descripcion_perfil": g["autor_descripcion"].values,
            "location_declarada": g["location"].values,
            "anotacion": "",
            "prediccion_modelo": "colombiano",
            "n_votos_colombia": g["n_votos_colombia"].values,
        }), "origen_geo", "origen_geo", log)
    else:
        log.warning("Falta %s: no se generan las muestras de polaridad ni de origen.",
                    cfg.PQ_ANALITICO.name)

    if cfg.PQ_ENTIDADES.exists():
        e = pd.read_parquet(cfg.PQ_ENTIDADES,
                            columns=["autor", "autor_nombre", "autor_descripcion",
                                     "entidad_autor"])
        e = e.drop_duplicates(subset=["autor"])
        e = e[e["autor_descripcion"].notna()
              & (e["autor_descripcion"].astype(str).str.strip() != "")]
        # sobremuestreo de las categorias sustantivas: son las que importa validar
        sust = e[e["entidad_autor"].isin(cfg.CATEGORIAS_TWITTER)]
        resto = e[~e["entidad_autor"].isin(cfg.CATEGORIAS_TWITTER)]
        n_s = min(len(sust), int(cfg.VALIDACION_N_ENTIDAD * 0.75))
        n_r = min(len(resto), cfg.VALIDACION_N_ENTIDAD - n_s)
        m = pd.concat([
            muestra_estratificada(sust, "entidad_autor", n_s, cfg.SEMILLA),
            resto.sample(n_r, random_state=cfg.SEMILLA) if n_r > 0 else resto.head(0),
        ]).sample(frac=1, random_state=cfg.SEMILLA)
        cat = pd.DataFrame([{"clave": k, "descripcion_en": v["etiqueta_en"]}
                            for k, v in cfg.CATEGORIAS.items()]
                           + [{"clave": cfg.SIN_CATEGORIA, "descripcion_en": "none of the above"}])
        escribir_muestra(pd.DataFrame({
            "id_muestra": np.arange(1, len(m) + 1),
            "autor": m["autor"].values,
            "nombre": m["autor_nombre"].values,
            "descripcion_perfil": m["autor_descripcion"].values,
            "anotacion": "",
            "prediccion_modelo": m["entidad_autor"].values,
        }), "entidad", "entidad", log, extra=cat)
    else:
        log.warning("Falta %s: no se genera la muestra de entidades.", cfg.PQ_ENTIDADES.name)

    # ---------------- 3. auditoria de cierre ----------------
    requisitos = [
        ("#3 Flujo iCARE con N por criterio", cfg.DIR_TABLAS / "s06_flujo_icare.csv"),
        ("#4 Cobertura temporal del corpus", cfg.DIR_TABLAS / "s01_cobertura_temporal.csv"),
        ("#5 Modelo de polaridad documentado", cfg.DIR_METRICAS / "s05_polaridad.json"),
        ("#6 Sensibilidad: una sola categoria",
         cfg.DIR_TABLAS / "s07_tabla3_sensibilidad_una_categoria.csv"),
        ("#7 Metricas de los 4 modelos clasicos", cfg.DIR_TABLAS / "s03b_metricas_modelos.csv"),
        ("#7 Metricas del 5o modelo (BETO)", cfg.DIR_TABLAS / "s03b_metricas_transformer.csv"),
        ("#8 Reglas de mapeo cuentas->categorias",
         cfg.DIR_TABLAS / "s04_resolucion_entidades.csv"),
        ("#8 Resolucion de menciones", cfg.DIR_TABLAS / "s04_resolucion_menciones.csv"),
        ("#9 Embudo del corpus", cfg.DIR_TABLAS / "s09_funnel_completo.csv"),
        ("#12/#13 Tabla 1 con Valid n", cfg.DIR_TABLAS / "s06_tabla1_sociodemograficas.csv"),
        ("#12/#13 Empleo como multi-respuesta",
         cfg.DIR_TABLAS / "s06_tabla1_empleo_multirespuesta.csv"),
        ("#16/#17 chi2 y V de Cramer", cfg.DIR_METRICAS / "s07_estadistica.json"),
        ("#16/#17 IC95% de negatividad", cfg.DIR_TABLAS / "s07_negatividad_por_entidad_ic95.csv"),
        ("#16/#17 Post-hoc con Bonferroni", cfg.DIR_TABLAS / "s07_posthoc_residuos.csv"),
        ("#19/#20 Concordancia cross-source",
         cfg.DIR_TABLAS / "s07_concordancia_cross_source.csv"),
        ("#10 Figura 1 arquitectura", cfg.DIR_FIGURAS / "figura_1_arquitectura.png"),
        ("#9 Figura 2 embudo", cfg.DIR_FIGURAS / "figura_2_embudo_corpus.png"),
        ("#14/#15 Figura 3 confianza", cfg.DIR_FIGURAS / "figura_3_confianza_icare.png"),
        ("#18 Figura 5 apiladas al 100%", cfg.DIR_FIGURAS / "figura_5_polaridad_por_entidad.png"),
        ("#19/#20 Figura 6 concordancia",
         cfg.DIR_FIGURAS / "figura_6_concordancia_cross_source.png"),
    ]
    aud = pd.DataFrame([{"requisito": r, "archivo": str(p_.relative_to(cfg.RAIZ)),
                         "estado": "LISTO" if p_.exists() else "FALTA"}
                        for r, p_ in requisitos])
    comun.guardar_tabla(aud, "s09_auditoria_cierre", log)
    faltan = aud[aud["estado"] == "FALTA"]
    log.info("\nAUDITORIA DE CIERRE: %d de %d insumos listos",
             len(aud) - len(faltan), len(aud))
    if len(faltan):
        log.warning("Faltan:\n%s", faltan[["requisito", "archivo"]].to_string(index=False))

    pendientes_humanos = [
        "Validacion humana de polaridad (muestra generada, sin anotar)",
        "Validacion humana de entidades (muestra generada, sin anotar)",
        "Validacion humana de origen geografico (muestra generada, sin anotar)",
        "Acuerdo interanotador: requiere un segundo anotador "
        "(config.VALIDACION_DOBLE_ANOTADOR = True)",
        "Filiaciones, autor de correspondencia y ORCID (#1)",
        "Comite de etica, numero de aprobacion y consentimiento iCARE (#11, #21)",
        "Data availability, Code availability, Funding, COI, CRediT (#22-#26)",
        "Verificacion manual de cada DOI y normalizacion bibliografica",
    ]
    log.info("\nPENDIENTES QUE EL CODIGO NO PUEDE RESOLVER:")
    for x in pendientes_humanos:
        log.info("  - %s", x)

    comun.guardar_metricas({
        "funnel": fun.to_dict("records") if not fun.empty else [],
        "auditoria": aud.to_dict("records"),
        "n_listos": int((aud["estado"] == "LISTO").sum()),
        "n_faltan": int(len(faltan)),
        "pendientes_no_automatizables": pendientes_humanos,
        "muestras_validacion": {
            "polaridad": cfg.VALIDACION_N_POLARIDAD,
            "entidad": cfg.VALIDACION_N_ENTIDAD,
            "origen_geo": cfg.VALIDACION_N_GEO,
            "doble_anotador": cfg.VALIDACION_DOBLE_ANOTADOR,
            "estado": "generadas, sin anotar",
        },
    }, "s09_consolidacion", log)

    log.info("ETAPA 07b COMPLETADA.")


if __name__ == "__main__":
    main()
