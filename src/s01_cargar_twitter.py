# -*- coding: utf-8 -*-
"""
ETAPA 01 — Carga, consolidacion y deduplicacion del corpus de Twitter/X.
VERSION LOCAL: lectura y escritura en streaming.

Diferencia con la version de APOLO
----------------------------------
Aquella leia los 290 archivos, guardaba los 290 DataFrames en una lista y
hacia un unico `pd.concat`. Con 1.44 millones de filas eso son unos 3 GB
en el momento del concat, y en una maquina de 16 GB con Windows encima es
arriesgado.

Esta version nunca tiene mas de un lote en memoria:

  1. Lee los .tab de LOTE_ARCHIVOS_TAB en LOTE_ARCHIVOS_TAB.
  2. De cada lote descarta lo que toque (sin fecha, retweets, idioma,
     fuera de ventana) y deduplica CONTRA TODO LO YA VISTO usando un
     conjunto de hashes de 64 bits de la clave (autor, texto, fecha).
     1.44M hashes ocupan ~120 MB: cabe de sobra, y evita tener que
     releer el texto completo para comparar.
  3. Escribe el lote depurado directamente al parquet de salida con un
     ParquetWriter incremental, en grupos de filas de PARQUET_ROW_GROUP.

El pico de memoria queda en torno a 300 MB en lugar de 3 GB, y el
resultado es identico: mismo N, mismos duplicados eliminados, mismas
estadisticas.

Que resuelve del informe del asesor
-----------------------------------
* Correccion #4 / seccion 2.3.3: "Resolver la aparente incompatibilidad
  entre '~5,000 tweets/day' y 1,442,058 tuits en el periodo informado."

  Los 290 archivos NO son uno por dia: son DOS recolecciones diarias sobre
  141 dias, con dos convenciones de nombre ("Covid-19_Spa_*" y
  "Covid19_Spa_*"), y cada archivo captura un bloque contiguo de ~5,000
  tuits que cubre solo una FRACCION del dia (1-2 horas), no el dia
  completo. De ahi que la suma de los archivos (~1.44 millones) sea el
  doble de lo que sugeriria "5,000 tuits/dia x 141 dias".

Salida
------
  intermedios/s01_twitter_crudo.parquet
  resultados/tablas/s01_inventario_archivos.csv
  resultados/tablas/s01_cobertura_temporal.csv
  resultados/tablas/s01_cobertura_por_dia.csv
  resultados/metricas/s01_carga.json
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
import comun

ETAPA = "s01"

COLUMNAS = [
    "Author", "Content", "Date", "Language", "Location",
    "Number of Likes", "Number of Retweets", "In Reply To",
    "Author Name", "Author Description",
    "Author Followers Count", "Author Verified",
]

RENOMBRE = {
    "Author": "autor", "Content": "texto", "Date": "fecha",
    "Language": "idioma", "Location": "location",
    "Number of Likes": "likes", "Number of Retweets": "retweets",
    "In Reply To": "responde_a", "Author Name": "autor_nombre",
    "Author Description": "autor_descripcion",
    "Author Followers Count": "autor_seguidores",
    "Author Verified": "autor_verificado",
}

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
         "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
         "noviembre": 11, "diciembre": 12}

# Algunos dias tienen una tercera recoleccion, cuyo archivo lleva el sufijo
# "_2" (p. ej. Covid19_Spa_5000_6Abril_2.tab). Sin ese sufijo opcional el
# patron no casaba con tres archivos: sus filas SI se parseaban, pero su
# cifra declarada no se sumaba, y el embudo salia con mas filas parseadas
# que declaradas, que es imposible en un filtro.
PATRON_NOMBRE = re.compile(
    r"^Covid-?19_Spa_(?P<declarado>\d+)_(?P<dia>\d+)(?P<mes>[A-Za-zñÑ]+)"
    r"(?:_(?P<repeticion>\d+))?\.tab$")


def metadatos_archivo(ruta: Path) -> dict:
    m = PATRON_NOMBRE.match(ruta.name)
    if not m:
        return {"archivo": ruta.name, "n_declarado": np.nan,
                "fecha_nominal": pd.NaT, "coleccion": "desconocida"}
    mes = MESES.get(m.group("mes").lower())
    fecha = (pd.Timestamp(year=2020, month=mes, day=int(m.group("dia")))
             if mes else pd.NaT)
    return {
        "archivo": ruta.name,
        "n_declarado": int(m.group("declarado")),
        "fecha_nominal": fecha,
        "coleccion": "A (Covid-19_Spa)" if ruta.name.startswith("Covid-19_") else "B (Covid19_Spa)",
    }


def leer_tab(ruta: Path, log) -> pd.DataFrame | None:
    """Fila 0 = nombres, fila 1 = tipos, fila 2 = metadatos. Las dos
    ultimas no son datos."""
    try:
        df = pd.read_csv(ruta, sep="\t", skiprows=[1, 2], dtype=str,
                         quotechar='"', on_bad_lines="skip", low_memory=False,
                         encoding="utf-8", encoding_errors="replace")
    except Exception as e:  # noqa: BLE001
        log.error("No se pudo leer %s: %s", ruta.name, e)
        return None
    for c in COLUMNAS:
        if c not in df.columns:
            log.warning("%s: falta la columna %s", ruta.name, c)
            df[c] = np.nan
    return df[COLUMNAS]


def clave_hash(df: pd.DataFrame) -> np.ndarray:
    """Hash de 64 bits de (autor, texto, fecha).

    Se usa el hash y no la tupla porque guardar 1.44M de tuplas de cadenas
    ocuparia varios GB, mientras que los hashes caben en ~120 MB. La
    probabilidad de colision con 1.4M de claves en un espacio de 2^64 es
    del orden de 10^-7: despreciable para deduplicar.
    """
    return pd.util.hash_pandas_object(
        df[["autor", "texto", "fecha"]].astype(str).agg("\x1f".join, axis=1),
        index=False).to_numpy(dtype=np.uint64)


def main() -> None:
    p = comun.parser_base(__doc__)
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if comun.debe_saltar(cfg.PQ_TWITTER_CRUDO, args.forzar, log):
        return

    archivos = sorted(cfg.DIR_TWITTER.glob("*.tab"))
    if args.prueba or cfg.MODO_PRUEBA:
        archivos = archivos[: cfg.N_PRUEBA_ARCHIVOS]
        log.warning("MODO PRUEBA: solo %d archivos", len(archivos))
    if not archivos:
        log.error("No hay archivos .tab en %s", cfg.DIR_TWITTER)
        sys.exit(1)
    log.info("Archivos .tab encontrados: %d", len(archivos))
    log.info("Lectura en streaming: lotes de %d archivos, sin cargar el corpus "
             "completo en memoria", cfg.LOTE_ARCHIVOS_TAB)

    ini = pd.Timestamp(cfg.TWITTER_FECHA_INICIO)
    fin = pd.Timestamp(cfg.TWITTER_FECHA_FIN) + pd.Timedelta(days=1)

    inventario = []
    vistos: set[int] = set()
    escritor = None
    esquema = None
    ruta_tmp = cfg.PQ_TWITTER_CRUDO.with_suffix(".tmp.parquet")

    n_leido = n_sin_fecha = n_rt = n_dup = n_no_es = n_fuera = n_final = 0
    dup_entre_colecciones = 0
    idiomas = Counter()
    por_dia_parcial = []

    try:
        with comun.Cronometro(log, f"procesando {len(archivos)} archivos"):
            for inicio in range(0, len(archivos), cfg.LOTE_ARCHIVOS_TAB):
                lote = archivos[inicio: inicio + cfg.LOTE_ARCHIVOS_TAB]
                piezas = []
                for ruta in lote:
                    meta = metadatos_archivo(ruta)
                    df = leer_tab(ruta, log)
                    if df is None:
                        meta["n_leido"] = 0
                        inventario.append(meta)
                        continue
                    meta["n_leido"] = len(df)
                    inventario.append(meta)
                    df = df.rename(columns=RENOMBRE)
                    df["archivo_origen"] = ruta.name
                    df["coleccion"] = meta["coleccion"]
                    piezas.append(df)
                if not piezas:
                    continue

                d = pd.concat(piezas, ignore_index=True)
                del piezas
                n_leido += len(d)

                # tipos
                d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
                for c in ("likes", "retweets", "autor_seguidores"):
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                d["texto"] = d["texto"].fillna("")

                sin_fecha = d["fecha"].isna()
                n_sin_fecha += int(sin_fecha.sum())
                d = d[~sin_fecha]

                es_rt = d["texto"].str.startswith("RT @")
                n_rt += int(es_rt.sum())
                d = d[~es_rt]

                # Deduplicacion global por hash.
                # Se consulta el conjunto elemento a elemento (O(1) cada
                # uno) en vez de reconstruir un array con np.isin en cada
                # lote, que seria O(n log n) sobre el millon largo de
                # hashes ya vistos y se repetiria quince veces.
                if len(d):
                    h = clave_hash(d)
                    conservar = np.empty(len(h), dtype=bool)
                    en_este_lote: set[int] = set()
                    repetidos_de_lotes_previos = 0
                    for i, x in enumerate(h.tolist()):
                        if x in vistos:
                            conservar[i] = False
                            repetidos_de_lotes_previos += 1
                        elif x in en_este_lote:
                            conservar[i] = False
                        else:
                            conservar[i] = True
                            en_este_lote.add(x)
                    n_dup += int((~conservar).sum())
                    dup_entre_colecciones += repetidos_de_lotes_previos
                    vistos.update(en_este_lote)
                    d = d[conservar]

                idiomas.update(d["idioma"].fillna("<vacio>").tolist())
                no_es = d["idioma"] != cfg.IDIOMA_OBJETIVO
                n_no_es += int(no_es.sum())
                d = d[~no_es]

                fuera = ~d["fecha"].between(ini, fin, inclusive="left")
                n_fuera += int(fuera.sum())
                d = d[~fuera]

                if not len(d):
                    continue

                por_dia_parcial.append(
                    d.assign(dia=d["fecha"].dt.date)
                     .groupby("dia")
                     .agg(n_tuits=("texto", "size"),
                          primera_hora=("fecha", "min"),
                          ultima_hora=("fecha", "max"),
                          archivos=("archivo_origen", lambda s: set(s)))
                     .reset_index())

                tabla = pa.Table.from_pandas(d.reset_index(drop=True),
                                             preserve_index=False)
                if escritor is None:
                    esquema = tabla.schema
                    escritor = pq.ParquetWriter(ruta_tmp, esquema,
                                                compression="snappy")
                else:
                    tabla = tabla.cast(esquema)
                escritor.write_table(tabla, row_group_size=cfg.PARQUET_ROW_GROUP)
                n_final += len(d)

                hechos = min(inicio + cfg.LOTE_ARCHIVOS_TAB, len(archivos))
                log.info("  %d/%d archivos — %s tuits conservados",
                         hechos, len(archivos), f"{n_final:,}")
                del d, tabla
    finally:
        if escritor is not None:
            escritor.close()

    if escritor is None:
        log.error("No se escribio ninguna fila. Revisar los datos de entrada.")
        sys.exit(1)
    import os
    os.replace(ruta_tmp, cfg.PQ_TWITTER_CRUDO)

    # ---------------- inventario y embudo ----------------
    inv = pd.DataFrame(inventario)
    comun.guardar_tabla(inv, "s01_inventario_archivos", log)
    n_declarado = int(inv["n_declarado"].sum())

    log.info("N declarado en los nombres de archivo: %s", f"{n_declarado:,}")
    log.info("N parseado: %s | sin fecha: %s | retweets: %s | duplicados: %s",
             f"{n_leido:,}", f"{n_sin_fecha:,}", f"{n_rt:,}", f"{n_dup:,}")
    log.info("  de los duplicados, %s ya se habian visto en lotes anteriores "
             "(solapamiento entre las dos recolecciones diarias)",
             f"{dup_entre_colecciones:,}")
    log.info("Descartados por idioma: %s | fuera de ventana: %s",
             f"{n_no_es:,}", f"{n_fuera:,}")

    for paso, n, desc in [
        ("E00_declarado", n_declarado,
         "Tuits declarados en los nombres de los archivos de recoleccion"),
        ("E01_parseado", n_leido, "Filas parseadas correctamente desde los .tab"),
        ("E02_con_fecha", n_leido - n_sin_fecha, "Filas con marca temporal valida"),
        ("E03_sin_retweets", n_leido - n_sin_fecha - n_rt,
         f"Retweets excluidos ({n_rt:,} encontrados)"),
        ("E04_deduplicado", n_leido - n_sin_fecha - n_rt - n_dup,
         f"Tras eliminar {n_dup:,} duplicados exactos por solapamiento entre recolecciones"),
        ("E05_espanol", n_leido - n_sin_fecha - n_rt - n_dup - n_no_es,
         f"Restringido a idioma='{cfg.IDIOMA_OBJETIVO}'"),
        ("E06_ventana", n_final,
         f"Dentro de la ventana {cfg.TWITTER_FECHA_INICIO} a {cfg.TWITTER_FECHA_FIN}"),
    ]:
        comun.registrar_funnel(ETAPA, paso, n, desc)

    # ---------------- cobertura temporal ----------------
    pd_ = pd.concat(por_dia_parcial, ignore_index=True)
    por_dia = (pd_.groupby("dia")
                  .agg(n_tuits=("n_tuits", "sum"),
                       primera_hora=("primera_hora", "min"),
                       ultima_hora=("ultima_hora", "max"),
                       archivos=("archivos", lambda s: set().union(*s)))
                  .reset_index())
    por_dia["n_recolecciones"] = por_dia["archivos"].map(len)
    por_dia["horas_distintas"] = np.nan
    por_dia = por_dia.drop(columns=["archivos"])
    por_dia["minutos_cubiertos"] = (
        (por_dia["ultima_hora"] - por_dia["primera_hora"]).dt.total_seconds() / 60).round(1)
    por_dia["pct_del_dia_cubierto"] = (por_dia["minutos_cubiertos"] / 1440 * 100).round(2)
    comun.guardar_tabla(por_dia, "s01_cobertura_por_dia", log)

    resumen = pd.DataFrame([{
        "dias_con_datos": int(por_dia["dia"].nunique()),
        "dias_en_ventana": int((pd.Timestamp(cfg.TWITTER_FECHA_FIN)
                                - pd.Timestamp(cfg.TWITTER_FECHA_INICIO)).days + 1),
        "archivos_de_recoleccion": len(archivos),
        "recolecciones_por_dia_mediana": float(por_dia["n_recolecciones"].median()),
        "tuits_por_dia_mediana": float(por_dia["n_tuits"].median()),
        "tuits_por_dia_min": int(por_dia["n_tuits"].min()),
        "tuits_por_dia_max": int(por_dia["n_tuits"].max()),
        "minutos_cubiertos_por_dia_mediana": float(por_dia["minutos_cubiertos"].median()),
        "pct_del_dia_cubierto_mediana": float(por_dia["pct_del_dia_cubierto"].median()),
    }])
    comun.guardar_tabla(resumen, "s01_cobertura_temporal", log)

    log.info("")
    log.info("COBERTURA TEMPORAL — cifras para la seccion 2.3.3 del manuscrito")
    log.info("  dias con datos ................ %s de %s en la ventana",
             resumen['dias_con_datos'][0], resumen['dias_en_ventana'][0])
    log.info("  recolecciones por dia (mediana)  %.1f",
             resumen['recolecciones_por_dia_mediana'][0])
    log.info("  tuits por dia (mediana) ....... %.0f", resumen['tuits_por_dia_mediana'][0])
    log.info("  minutos del dia cubiertos ..... %.0f min (%.1f%% del dia)",
             resumen['minutos_cubiertos_por_dia_mediana'][0],
             resumen['pct_del_dia_cubierto_mediana'][0])
    log.info("  -> La cobertura intradiaria es PARCIAL. Debe declararse en")
    log.info("     Methods y en Limitations.")
    log.info("")

    comun.guardar_metricas({
        "modo": "streaming (version local)",
        "archivos": len(archivos),
        "lote_archivos": cfg.LOTE_ARCHIVOS_TAB,
        "n_declarado": n_declarado,
        "n_parseado": n_leido,
        "n_sin_fecha_descartadas": n_sin_fecha,
        "n_retweets_encontrados": n_rt,
        "n_duplicados_eliminados": n_dup,
        "n_duplicados_entre_recolecciones": dup_entre_colecciones,
        "n_descartados_por_idioma": n_no_es,
        "n_fuera_de_ventana": n_fuera,
        "conteo_por_idioma": dict(idiomas.most_common(10)),
        "n_final": n_final,
        "cobertura": resumen.to_dict("records")[0],
    }, "s01_carga", log)

    mb = cfg.PQ_TWITTER_CRUDO.stat().st_size / 1e6
    log.info("Guardado %s (%s filas, %.1f MB)",
             cfg.PQ_TWITTER_CRUDO.name, f"{n_final:,}", mb)
    log.info("ETAPA 01 COMPLETADA. Corpus consolidado: %s tuits", f"{n_final:,}")


if __name__ == "__main__":
    main()
