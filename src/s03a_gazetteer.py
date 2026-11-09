# -*- coding: utf-8 -*-
"""
ETAPA 03a — Origen colombiano, evidencia 1: gazetteer + lexicon + geoetiqueta.

Que resuelve del informe del asesor
-----------------------------------
* Seccion 2.3.5: "Documentar los cinco modelos, parametros esenciales,
  conjunto de entrenamiento, variable objetivo, validacion, metricas y
  forma de combinar el GAZETTEER, LEXICON y grafo con el clasificador
  automatico."

  Esta es la primera de las tres evidencias. Es puramente basada en
  reglas: no entrena nada y por tanto no tiene metricas de validacion
  cruzada. Su papel en la regla de combinacion (etapa 03d) es el de un
  voto, no el de un criterio unico.

CAMBIO IMPORTANTE RESPECTO DE LA VERSION 1
------------------------------------------
La version 1 trataba los ~1,500 terminos del gazetteer como una lista
plana. Medido sobre este corpus, de los veinte terminos mas activados
solo cinco eran inequivocamente colombianos; el resto eran palabras
comunes ("fundacion", "une", "cesar", "toca"), apellidos que tambien son
municipios ("gonzalez", "ortega") y lugares mas conocidos de otros paises
("madrid", "santiago", "california", "florida"). En la jerga, "china" y
sus variantes sumaban 396 activaciones referidas casi siempre a China el
pais, no al sentido colombiano de "nino".

Ahora el gazetteer tiene dos niveles (ver recursos/gazetteer.py):

  NUCLEO      pais, departamentos, capitales y ciudades principales,
              menos los ambiguos. Coincidir aqui es evidencia por si sola.
  EXTENDIDO   municipios, pueblos patrimonio y parques nacionales. Miles
              de nombres, muchos homonimos. Coincidir aqui NO basta:
              exige otra senal (nucleo, jerga o Location = 'CO').

Definicion operativa
--------------------
    senal_fuerte = coincide_nucleo O coincide_jerga
    senal_debil  = coincide_extendido

    Seed_1 = (senal_fuerte O (senal_debil Y (senal_fuerte O location_co)))
             Y location_compatible

  location_compatible: Location del perfil es 'CO' o esta vacio. Excluye
  a quien se declara de otro pais; sin este componente, cualquier tuit
  mexicano que nombre a Bogota entraria al corpus.

Poner config.GAZETTEER_EXTENDIDO_EXIGE_CORROBORACION = False reproduce el
comportamiento plano de la version 1, para poder comparar.

Salida
------
  intermedios/s03a_seed_gazetteer.parquet
  resultados/tablas/s03a_componentes_gazetteer.csv
  resultados/tablas/s03a_terminos_mas_frecuentes.csv   <- REVISAR A MANO
  resultados/metricas/s03a_gazetteer.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "recursos"))

import config as cfg
import comun
from s02_preprocesar import quitar_acentos

ETAPA = "s03a"

# [LOCAL] El corpus se recorre por bloques. La version de APOLO guardaba
# las listas de coincidencias de los 1.44M de tuits (tres columnas de
# listas de Python, mas de 1 GB); aqui solo se conservan los vectores
# booleanos y los contadores, que ocupan unos pocos MB.


def normalizar_lista(lista, excluir: set, longitud_minima: int = 0) -> list[str]:
    """Minusculas, sin tildes, sin duplicados, sin los ambiguos.

    Se ordena por longitud descendente para que en la busqueda gane la
    coincidencia mas larga ('norte de santander' antes que 'santander').
    La longitud minima solo se aplica a terminos de una sola palabra: los
    compuestos son inequivocos por su propia extension.
    """
    vistos = set()
    for x in lista:
        t = quitar_acentos(str(x).lower()).strip()
        if not t or t in excluir:
            continue
        if longitud_minima and " " not in t and len(t) < longitud_minima:
            continue
        vistos.add(t)
    return sorted(vistos, key=len, reverse=True)


def compilar_patron(terminos: list[str]) -> re.Pattern | None:
    if not terminos:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terminos) + r")\b")


def buscar(serie: pd.Series, patron):
    if patron is None:
        return pd.Series([[] for _ in range(len(serie))], index=serie.index)
    return serie.map(lambda s: patron.findall(s) if s else [])


def main() -> None:
    p = comun.parser_base(__doc__)
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if comun.debe_saltar(cfg.PQ_SEED1, args.forzar, log):
        return
    if not cfg.PQ_TWITTER_PREP.exists():
        log.error("Falta %s. Ejecutar antes la etapa 02.", cfg.PQ_TWITTER_PREP.name)
        sys.exit(1)

    import gazetteer as gz

    excl_top = cfg.TOPONIMOS_AMBIGUOS_EXCLUIDOS
    excl_jer = cfg.JERGA_AMBIGUA_EXCLUIDA

    nucleo = normalizar_lista(gz.GAZETTEER_NUCLEO, excl_top)
    extendido = normalizar_lista(gz.GAZETTEER_EXTENDIDO, excl_top,
                                 cfg.GAZETTEER_LONGITUD_MINIMA_EXTENDIDO)
    extendido = [t for t in extendido if t not in set(nucleo)]
    jerga = normalizar_lista(gz.JERGA, excl_jer)

    log.info("Gazetteer NUCLEO:    %4d terminos (de %d antes de excluir ambiguos)",
             len(nucleo), len(set(gz.GAZETTEER_NUCLEO)))
    log.info("Gazetteer EXTENDIDO: %4d terminos (de %d; se quitan ambiguos y los de "
             "menos de %d caracteres)", len(extendido), len(set(gz.GAZETTEER_EXTENDIDO)),
             cfg.GAZETTEER_LONGITUD_MINIMA_EXTENDIDO)
    log.info("Lexicon JERGA:       %4d terminos (de %d)", len(jerga), len(set(gz.JERGA)))

    pat_nuc = compilar_patron(nucleo)
    pat_ext = compilar_patron(extendido)
    pat_jer = compilar_patron(jerga)

    import pyarrow.parquet as pq
    fuente = pq.ParquetFile(cfg.PQ_TWITTER_PREP)
    n = fuente.metadata.num_rows
    n_bloques = int(np.ceil(n / cfg.CHUNK_TWEETS))
    log.info("Evaluando %s tuits en %d bloques", f"{n:,}", n_bloques)

    partes_nuc, partes_ext, partes_jer = [], [], []
    partes_co, partes_vacia = [], []
    conteos = {"nucleo": Counter(), "extendido": Counter(), "jerga": Counter()}
    pendientes = []   # (bloque_de_hallazgos, mascara_seed1) para los contadores

    with comun.Cronometro(log, "buscando toponimos y jerga"):
        for i, lote in enumerate(fuente.iter_batches(
                batch_size=cfg.CHUNK_TWEETS,
                columns=["texto_normalizado", "location"])):
            d = lote.to_pandas()
            hn = buscar(d["texto_normalizado"], pat_nuc)
            he = buscar(d["texto_normalizado"], pat_ext)
            hj = buscar(d["texto_normalizado"], pat_jer)
            cn, ce, cj = hn.str.len() > 0, he.str.len() > 0, hj.str.len() > 0

            l = d["location"].fillna("").str.strip().str.upper()
            lco, lvac = (l == "CO"), (l == "")

            # se calcula seed_1 del bloque para contar solo los terminos
            # que activaron tuits efectivamente seleccionados
            fuerte = cn | cj
            if cfg.GAZETTEER_EXTENDIDO_EXIGE_CORROBORACION:
                senal = fuerte | (ce & (fuerte | lco))
            else:
                senal = fuerte | ce
            sel = (senal & (lco | lvac)).to_numpy()
            for etiqueta, hall in (("nucleo", hn), ("extendido", he), ("jerga", hj)):
                for lst in hall[sel]:
                    conteos[etiqueta].update(lst)

            partes_nuc.append(cn.to_numpy()); partes_ext.append(ce.to_numpy())
            partes_jer.append(cj.to_numpy())
            partes_co.append(lco.to_numpy()); partes_vacia.append(lvac.to_numpy())
            del d, hn, he, hj
            if (i + 1) % 5 == 0 or (i + 1) == n_bloques:
                log.info("  bloque %d/%d", i + 1, n_bloques)

    c_nuc = pd.Series(np.concatenate(partes_nuc))
    c_ext = pd.Series(np.concatenate(partes_ext))
    c_jer = pd.Series(np.concatenate(partes_jer))
    loc_co = pd.Series(np.concatenate(partes_co))
    loc_vacia = pd.Series(np.concatenate(partes_vacia))
    del partes_nuc, partes_ext, partes_jer, partes_co, partes_vacia
    loc_otro = (~loc_co) & (~loc_vacia)
    loc_compatible = loc_co | loc_vacia

    senal_fuerte = c_nuc | c_jer
    if cfg.GAZETTEER_EXTENDIDO_EXIGE_CORROBORACION:
        senal = senal_fuerte | (c_ext & (senal_fuerte | loc_co))
        regla = ("(nucleo O jerga) O (extendido Y (nucleo O jerga O Location=='CO')), "
                 "todo ello Y Location compatible")
    else:
        senal = senal_fuerte | c_ext
        regla = "(nucleo O extendido O jerga) Y Location compatible  [modo plano v1]"
    seed1 = senal & loc_compatible

    componentes = pd.DataFrame([
        {"componente": "coincide_gazetteer_NUCLEO", "n": int(c_nuc.sum())},
        {"componente": "coincide_gazetteer_EXTENDIDO", "n": int(c_ext.sum())},
        {"componente": "coincide_lexicon_jerga", "n": int(c_jer.sum())},
        {"componente": "senal_fuerte (nucleo o jerga)", "n": int(senal_fuerte.sum())},
        {"componente": "solo_extendido_sin_corroborar",
         "n": int((c_ext & ~senal_fuerte & ~loc_co).sum())},
        {"componente": "location_declara_CO", "n": int(loc_co.sum())},
        {"componente": "location_vacia", "n": int(loc_vacia.sum())},
        {"componente": "location_declara_otro_pais", "n": int(loc_otro.sum())},
        {"componente": "SEED_1_final", "n": int(seed1.sum())},
    ])
    componentes["pct_del_corpus"] = (componentes["n"] / n * 100).round(2)
    comun.guardar_tabla(componentes, "s03a_componentes_gazetteer", log)
    log.info("\n%s", componentes.to_string(index=False))

    descartados_extendido = int((c_ext & ~senal_fuerte & ~loc_co).sum())
    log.info("Tuits que la version 1 habria marcado y aqui NO, por apoyarse solo en "
             "un municipio sin corroborar: %s", f"{descartados_extendido:,}")
    perdidos = int((senal & loc_otro).sum())
    log.info("Coincidian en texto pero declaran otro pais (excluidos): %s", f"{perdidos:,}")

    # terminos activados EN LOS SELECCIONADOS: es el control de calidad barato.
    # Se acumularon bloque a bloque durante el recorrido de arriba.
    terminos = pd.concat(
        [pd.DataFrame(c.most_common(150), columns=["termino", "frecuencia"]).assign(tipo=k)
         for k, c in conteos.items() if c], ignore_index=True)
    comun.guardar_tabla(terminos, "s03a_terminos_mas_frecuentes", log)
    for k, c in conteos.items():
        log.info("Mas activados (%s): %s", k, c.most_common(10))
    log.info("REVISAR s03a_terminos_mas_frecuentes.csv sobre el corpus completo: si "
             "aparecen palabras comunes o lugares de otros paises, anadirlas a "
             "config.TOPONIMOS_AMBIGUOS_EXCLUIDOS y relanzar con --forzar.")

    salida = pd.DataFrame({
        "gaz_nucleo": c_nuc.values,
        "gaz_extendido": c_ext.values,
        "gaz_jerga": c_jer.values,
        "loc_co": loc_co.values,
        "loc_vacia": loc_vacia.values,
        "loc_otro_pais": loc_otro.values,
        "seed_1": seed1.values,
    })
    comun.guardar_parquet(salida, cfg.PQ_SEED1, log)

    comun.registrar_funnel(
        ETAPA, "E08_seed_gazetteer", int(seed1.sum()),
        "Evidencia 1: gazetteer en dos niveles + lexicon + Location (voto, no criterio final)")

    comun.guardar_metricas({
        "regla": regla,
        "extendido_exige_corroboracion": cfg.GAZETTEER_EXTENDIDO_EXIGE_CORROBORACION,
        "longitud_minima_extendido": cfg.GAZETTEER_LONGITUD_MINIMA_EXTENDIDO,
        "n_terminos": {"nucleo": len(nucleo), "extendido": len(extendido), "jerga": len(jerga)},
        "n_excluidos_por_ambiguedad": {"toponimos": len(excl_top), "jerga": len(excl_jer)},
        "componentes": componentes.to_dict("records"),
        "descartados_por_exigir_corroboracion": descartados_extendido,
        "excluidos_por_otro_pais": perdidos,
        "n_seed_1": int(seed1.sum()),
        "pct_seed_1": round(100 * float(seed1.mean()), 2),
        "terminos_mas_activados": {k: c.most_common(25) for k, c in conteos.items()},
        "advertencia": ("las listas de exclusion son revisables; revisar "
                        "s03a_terminos_mas_frecuentes.csv sobre el corpus completo"),
    }, "s03a_gazetteer", log)

    log.info("ETAPA 03a COMPLETADA. Seed_1 = %s tuits (%.2f%%)",
             f"{int(seed1.sum()):,}", 100 * float(seed1.mean()))


if __name__ == "__main__":
    main()
