# -*- coding: utf-8 -*-
"""
ETAPA 02 — Preprocesamiento del texto. VERSION LOCAL: por bloques.

Diferencia con la version de APOLO
----------------------------------
Aquella cargaba el corpus entero en un DataFrame y le anadia tres
representaciones del texto mas las listas de menciones y hashtags. Sobre
1.44M de tuits eso ronda los 6 GB, inviable en una maquina de 16 GB con
Windows.

Esta version lee el parquet de la etapa 01 POR BLOQUES de
config.CHUNK_TWEETS filas, procesa cada bloque completo (extraccion,
normalizacion y lematizacion) y lo escribe a disco antes de pasar al
siguiente. Al terminar une los bloques en streaming, sin cargarlos a la
vez. El pico de memoria es el de un bloque: unos 400 MB.

Los bloques quedan en intermedios/s02_bloques/, asi que si se interrumpe
el proceso se reanuda desde el ultimo bloque completo.

Que resuelve del informe del asesor
-----------------------------------
* Seccion 2.3.4: "No puede afirmarse que se eliminan mentions y luego
  analizar entidades citadas sin explicar el orden. Indicar que
  @mentions/handles y entidades se extraen ANTES de la limpieza
  linguistica o se preservan en campos separados."

  Este script hace exactamente eso, y en ese orden:
    1. Del texto CRUDO se extraen menciones, hashtags y URLs a columnas
       propias (`menciones`, `hashtags`, `urls`). Nada se pierde.
    2. Solo despues se construyen las dos representaciones limpias.

  Las tres representaciones del texto conviven en el parquet de salida y
  cada etapa posterior declara cual usa:

    texto              texto original, intacto. Lo usa el modelo de
                       polaridad (etapa 05), que fue entrenado sobre
                       tuits crudos y se degrada si se le quita la
                       puntuacion, los emojis o las mayusculas.
    texto_normalizado  minusculas, sin tildes, sin URLs, CONSERVANDO @ y #.
                       Lo usa el gazetteer (etapa 03a).
    texto_limpio       lematizado, sin stopwords, sin menciones ni
                       hashtags. Lo usan los modelos ML de bolsa de
                       palabras (etapa 03b).

Reanudacion
-----------
La lematizacion con spaCy sobre ~1.4M tuits es la parte lenta del
pipeline. Se procesa por bloques de config.CHUNK_TWEETS y cada bloque se
guarda en intermedios/s02_bloques/. Si el job se cae o se queda sin
tiempo de pared, al relanzarlo continua desde el ultimo bloque completo.

Salida
------
  intermedios/s02_twitter_preprocesado.parquet
  resultados/tablas/s02_top_menciones.csv
  resultados/tablas/s02_top_hashtags.csv
  resultados/metricas/s02_preprocesamiento.json
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
import comun

ETAPA = "s02"

DIR_BLOQUES = cfg.DIR_INTERMEDIOS / "s02_bloques"

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENCION = re.compile(r"@[A-Za-z0-9_]{1,15}")
RE_HASHTAG = re.compile(r"#[\wáéíóúñÁÉÍÓÚÑ]+", re.UNICODE)
RE_NO_ALFA = re.compile(r"[^a-zñ\s@#]")
RE_ESPACIOS = re.compile(r"\s+")


def quitar_acentos(s: str) -> str:
    """Quita tildes y dieresis conservando la enye."""
    s = s.replace("ñ", "\x00").replace("Ñ", "\x01")
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return s.replace("\x00", "ñ").replace("\x01", "Ñ")


def extraer_entidades_superficiales(texto: str) -> tuple[list, list, list]:
    """PASO 1: sobre el texto CRUDO, antes de cualquier limpieza.

    Devuelve (menciones, hashtags, urls). Las menciones se normalizan a
    minusculas porque los handles de Twitter no distinguen mayusculas y
    la etapa 04 las cruza contra el diccionario de cuentas conocidas.
    """
    if not isinstance(texto, str) or not texto:
        return [], [], []
    urls = RE_URL.findall(texto)
    sin_url = RE_URL.sub(" ", texto)
    menciones = [m.lower() for m in RE_MENCION.findall(sin_url)]
    hashtags = [h.lower() for h in RE_HASHTAG.findall(sin_url)]
    return menciones, hashtags, urls


def normalizar(texto: str) -> str:
    """PASO 2a: minusculas, sin tildes, sin URLs. Conserva @ y # porque
    el gazetteer se aplica sobre esta version y algunos toponimos
    aparecen dentro de hashtags (#Bogota, #Medellin)."""
    if not isinstance(texto, str) or not texto:
        return ""
    s = RE_URL.sub(" ", texto.lower())
    s = quitar_acentos(s)
    s = RE_NO_ALFA.sub(" ", s)
    return RE_ESPACIOS.sub(" ", s).strip()


def cargar_spacy(log):
    import spacy
    try:
        nlp = spacy.load("es_core_news_sm",
                         disable=["parser", "ner", "textcat", "attribute_ruler"])
    except OSError:
        log.error("Falta el modelo de spaCy. Instalarlo con:")
        log.error("  python -m spacy download es_core_news_sm")
        log.error("  (o, en Windows:  correr.bat reparar-spacy)")
        sys.exit(1)
    except ValueError as e:
        # "int is not allowed for map key when strict_map_key=True" es el
        # sintoma tipico de un desajuste entre la version de spaCy y la del
        # modelo: el .msgpack del modelo no lo entiende el spaCy instalado.
        # Pasa al mezclar el spaCy de conda-forge con un modelo bajado
        # aparte. El traceback de msgpack no dice nada de esto, asi que se
        # traduce aqui.
        log.error("spaCy no pudo cargar el modelo: %s", e)
        log.error("")
        log.error("Es un desajuste de versiones entre spaCy y es_core_news_sm.")
        log.error("Solucion:")
        log.error("  correr.bat reparar-spacy")
        log.error("o a mano, con el entorno activado:")
        log.error("  pip uninstall -y spacy srsly thinc es-core-news-sm")
        log.error("  pip install -U spacy")
        log.error("  python -m spacy download es_core_news_sm")
        log.error("")
        log.error("Comprobar despues con:  python -m spacy validate")
        sys.exit(1)
    nlp.max_length = 2_000_000
    return nlp


# Lista de respaldo. El pipeline NO puede depender de una descarga de red
# en tiempo de ejecucion: si el nodo de computo no tiene salida a internet
# la etapa moriria a mitad. Es la lista de stopwords en espanol de NLTK,
# embebida para que el resultado sea identico con o sin red.
STOPWORDS_ES_RESPALDO = """
a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
durante e el ella ellas ellos en entre era erais eramos eran eras eres es esa esas
ese eso esos esta estaba estabais estabamos estaban estabas estad estada estadas
estado estados estais estamos estan estando estar estara estaran estaras estare
estareis estaremos estaria estariais estariamos estarian estarias estas este esteis
estemos esten estes esto estos estoy estuve estuviera estuvierais estuvieramos
estuvieran estuvieras estuvieron estuviese estuvieseis estuviesemos estuviesen
estuvieses estuvimos estuviste estuvisteis estuvo fue fuera fuerais fueramos fueran
fueras fueron fuese fueseis fuesemos fuesen fueses fui fuimos fuiste fuisteis ha
habeis habia habiais habiamos habian habias habida habidas habido habidos habiendo
habra habran habras habre habreis habremos habria habriais habriamos habrian habrias
han has hasta hay haya hayais hayamos hayan hayas he hemos hube hubiera hubierais
hubieramos hubieran hubieras hubieron hubiese hubieseis hubiesemos hubiesen hubieses
hubimos hubiste hubisteis hubo la las le les lo los mas me mi mia mias mio mios mis
mucho muchos muy nada ni no nos nosotras nosotros nuestra nuestras nuestro nuestros
o os otra otras otro otros para pero poco por porque que quien quienes se sea seais
seamos sean seas sentid sentida sentidas sentido sentidos sera seran seras sere
sereis seremos seria seriais seriamos serian serias si sido siente sin sintiendo
sobre sois somos son soy su sus suya suyas suyo suyos también tanto te tendra
tendran tendras tendre tendreis tendremos tendria tendriais tendriamos tendrian
tendrias tened teneis tenemos tenga tengais tengamos tengan tengas tengo tenia
teniais teniamos tenian tenias tenida tenidas tenido tenidos teniendo ti tiene
tienen tienes todo todos tu tus tuve tuviera tuvierais tuvieramos tuvieran tuvieras
tuvieron tuviese tuvieseis tuviesemos tuviesen tuvieses tuvimos tuviste tuvisteis
tuvo tuya tuyas tuyo tuyos un una uno unos vosotras vosotros vuestra vuestras
vuestro vuestros y ya yo el mi que se si mas
""".split()


def cargar_stopwords(log) -> set:
    """Intenta NLTK; si no esta disponible usa la lista embebida.

    En ambos casos las stopwords se normalizan igual que el texto (sin
    tildes, minusculas) para que los emparejamientos casen.
    """
    sw = None
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("spanish"))
        log.info("Stopwords: NLTK (%d terminos)", len(sw))
    except Exception:  # noqa: BLE001
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            sw = set(stopwords.words("spanish"))
            log.info("Stopwords: NLTK tras descarga (%d terminos)", len(sw))
        except Exception:  # noqa: BLE001
            sw = None
    if not sw:
        sw = set(STOPWORDS_ES_RESPALDO)
        log.warning("NLTK no disponible. Se usa la lista de stopwords embebida "
                    "(%d terminos). El resultado es equivalente.", len(sw))
    return {quitar_acentos(w.lower()) for w in sw}


def lematizar_bloque(textos, nlp, stop, n_proc: int) -> list[str]:
    """PASO 2b: lematiza y filtra. Se ejecuta sobre texto ya normalizado,
    de modo que las menciones y hashtags ya fueron extraidos aparte."""
    salida = []
    for doc in nlp.pipe(textos, batch_size=1000, n_process=n_proc):
        toks = [t.lemma_ for t in doc
                if len(t.lemma_) > 2 and t.lemma_ not in stop
                and not t.lemma_.startswith(("@", "#"))]
        salida.append(" ".join(toks))
    return salida


def procesar_bloque(d: pd.DataFrame, nlp, stop, lematizar: bool) -> pd.DataFrame:
    """Aplica a un bloque el mismo tratamiento que la version de APOLO
    aplicaba al corpus completo, en el mismo orden."""
    # PASO 1: extraccion sobre el texto CRUDO
    extraidos = d["texto"].map(extraer_entidades_superficiales)
    d["menciones"] = [e[0] for e in extraidos]
    d["hashtags"] = [e[1] for e in extraidos]
    d["urls"] = [e[2] for e in extraidos]
    del extraidos
    d["n_menciones"] = d["menciones"].str.len()
    d["n_hashtags"] = d["hashtags"].str.len()

    # PASO 2a: normalizacion
    d["texto_normalizado"] = d["texto"].map(normalizar)

    # PASO 2b: lematizacion
    if lematizar:
        d["texto_limpio"] = lematizar_bloque(
            d["texto_normalizado"].tolist(), nlp, stop, cfg.N_PROCESOS)
    else:
        d["texto_limpio"] = (d["texto_normalizado"]
                             .str.replace(r"[@#]\w+", " ", regex=True)
                             .str.replace(r"\s+", " ", regex=True).str.strip())
    d["texto_limpio"] = d["texto_limpio"].fillna("")
    return d


def main() -> None:
    p = comun.parser_base(__doc__)
    p.add_argument("--sin-lematizar", action="store_true",
                   help="Salta spaCy y usa el texto normalizado como texto_limpio. "
                        "Util para una pasada rapida de diagnostico.")
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()
    DIR_BLOQUES.mkdir(parents=True, exist_ok=True)

    if comun.debe_saltar(cfg.PQ_TWITTER_PREP, args.forzar, log):
        return
    if not cfg.PQ_TWITTER_CRUDO.exists():
        log.error("Falta %s. Ejecutar antes la etapa 01.", cfg.PQ_TWITTER_CRUDO.name)
        sys.exit(1)

    import pyarrow as pa
    import pyarrow.parquet as pq

    fuente = pq.ParquetFile(cfg.PQ_TWITTER_CRUDO)
    n_total = fuente.metadata.num_rows
    n_bloques = int(np.ceil(n_total / cfg.CHUNK_TWEETS))
    log.info("Corpus de entrada: %s tuits — %d bloques de %s (reanudable)",
             f"{n_total:,}", n_bloques, f"{cfg.CHUNK_TWEETS:,}")

    nlp = stop = None
    if not args.sin_lematizar:
        nlp = cargar_spacy(log)
        stop = cargar_stopwords(log)
    else:
        log.warning("--sin-lematizar activo. NO usar para los resultados definitivos.")

    # ---------------- procesar bloque a bloque ----------------
    cont_men, cont_hash = Counter(), Counter()
    n_con_mencion = n_menciones_tot = 0

    b = 0
    for lote in fuente.iter_batches(batch_size=cfg.CHUNK_TWEETS):
        destino = DIR_BLOQUES / f"bloque_{b:04d}.parquet"
        if destino.exists() and not args.forzar:
            log.info("  bloque %d/%d ya existe, se salta", b + 1, n_bloques)
            d = pd.read_parquet(destino, columns=["menciones", "hashtags", "n_menciones"])
        else:
            with comun.Cronometro(log, f"bloque {b + 1}/{n_bloques}"):
                d = procesar_bloque(lote.to_pandas(), nlp, stop, not args.sin_lematizar)
                d.to_parquet(destino, index=False)
        for lista in d["menciones"]:
            cont_men.update(lista)
        for lista in d["hashtags"]:
            cont_hash.update(lista)
        n_con_mencion += int((d["n_menciones"] > 0).sum())
        n_menciones_tot += int(d["n_menciones"].sum())
        del d
        b += 1

    # ---------------- unir los bloques en streaming ----------------
    bloques = sorted(DIR_BLOQUES.glob("bloque_*.parquet"))
    log.info("Uniendo %d bloques sin cargarlos todos a la vez", len(bloques))
    ruta_tmp = cfg.PQ_TWITTER_PREP.with_suffix(".tmp.parquet")
    escritor = esquema = None
    n_escritas = n_vacios = 0
    try:
        for f in bloques:
            t = pq.read_table(f)
            if escritor is None:
                esquema = t.schema
                escritor = pq.ParquetWriter(ruta_tmp, esquema, compression="snappy")
            else:
                t = t.cast(esquema)
            escritor.write_table(t, row_group_size=cfg.PARQUET_ROW_GROUP)
            n_escritas += t.num_rows
            col = t.column("texto_limpio").to_pylist()
            n_vacios += sum(1 for x in col if not x or not x.strip())
            del t
    finally:
        if escritor is not None:
            escritor.close()

    if n_escritas != n_total:
        log.error("Los bloques suman %s filas y el corpus tiene %s. Borrar "
                  "%s y relanzar con --forzar.",
                  f"{n_escritas:,}", f"{n_total:,}", DIR_BLOQUES)
        sys.exit(1)

    import os
    os.replace(ruta_tmp, cfg.PQ_TWITTER_PREP)

    # ---------------- reporte ----------------
    log.info("Tuits con al menos una mencion: %s (%.1f%%)",
             f"{n_con_mencion:,}", 100 * n_con_mencion / max(n_total, 1))
    log.info("Menciones distintas: %s | Hashtags distintos: %s",
             f"{len(cont_men):,}", f"{len(cont_hash):,}")
    log.info("Tuits con texto_limpio vacio: %s (%.2f%%)",
             f"{n_vacios:,}", 100 * n_vacios / max(n_total, 1))

    comun.guardar_tabla(
        pd.DataFrame(cont_men.most_common(500), columns=["mencion", "frecuencia"]),
        "s02_top_menciones", log)
    comun.guardar_tabla(
        pd.DataFrame(cont_hash.most_common(500), columns=["hashtag", "frecuencia"]),
        "s02_top_hashtags", log)

    comun.registrar_funnel(ETAPA, "E07_preprocesado", n_total,
                           "Menciones y hashtags extraidos antes de la limpieza linguistica")

    comun.guardar_metricas({
        "modo": "por bloques (version local)",
        "tamano_bloque": cfg.CHUNK_TWEETS,
        "n_bloques": len(bloques),
        "n_tuits": n_total,
        "n_con_mencion": n_con_mencion,
        "pct_con_mencion": round(100 * n_con_mencion / max(n_total, 1), 2),
        "menciones_distintas": len(cont_men),
        "hashtags_distintos": len(cont_hash),
        "menciones_totales": n_menciones_tot,
        "menciones_por_tuit_media": round(n_menciones_tot / max(n_total, 1), 3),
        "texto_limpio_vacio": n_vacios,
        "lematizado": not args.sin_lematizar,
        "top20_menciones": cont_men.most_common(20),
        "top20_hashtags": cont_hash.most_common(20),
    }, "s02_preprocesamiento", log)

    mb = cfg.PQ_TWITTER_PREP.stat().st_size / 1e6
    log.info("Guardado %s (%s filas, %.1f MB)",
             cfg.PQ_TWITTER_PREP.name, f"{n_total:,}", mb)
    log.info("ETAPA 02 COMPLETADA.")


if __name__ == "__main__":
    main()
