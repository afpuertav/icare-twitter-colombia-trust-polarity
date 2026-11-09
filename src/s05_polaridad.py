# -*- coding: utf-8 -*-
"""
ETAPA 05 — Polaridad del discurso.

Que resuelve del informe del asesor
-----------------------------------
* Correccion #5 / seccion 2.3.7: "Identificar modelo exacto, version,
  idioma/corpus de entrenamiento, procedimiento de inferencia y
  validacion manual. Si se uso pysentimiento/RoBERTuito, declararlo solo
  si corresponde realmente al pipeline ejecutado."

  Modelo efectivamente ejecutado, para declarar en Methods:

    Nombre     cardiffnlp/twitter-xlm-roberta-base-sentiment
    Arquitectura XLM-RoBERTa base (270M parametros)
    Preentrenamiento ~198M tuits multilingues
    Ajuste fino   corpus de analisis de sentimiento en 8 idiomas
                  (incluye espanol), etiquetas Negative/Neutral/Positive
    Salida        3 clases + probabilidad por clase
    Inferencia    truncado a 128 tokens, lotes de 128, softmax sobre logits

  NO se usa pysentimiento ni RoBERTuito. El manuscrito no debe citarlos.
  La version 1 del pipeline si empleaba pysentimiento, pero para EMOCION
  (polaridad.py usaba cardiffnlp; sentimientos.py usaba pysentimiento para
  ira/asco/miedo/alegria/tristeza/sorpresa). La emocion no forma parte de
  este manuscrito y por eso no se recalcula.

* Correccion #6 / seccion 2.3.7: "Indicar si la polaridad es global del
  tuit o especifica de la entidad."

  DECISION TOMADA Y QUE DEBE QUEDAR ESCRITA: la polaridad es GLOBAL DEL
  TUIT. No es target-dependent ni aspect-based. En consecuencia el
  manuscrito NO puede escribir "sentiment toward an entity"; la
  formulacion admisible es "polarity of the discourse in which the entity
  is mentioned". config.POLARIDAD_ES_TARGET_DEPENDENT = False deja la
  decision registrada en codigo.

  Mitigacion: la etapa 07 repite el analisis restringido a los tuits que
  mencionan UNA SOLA categoria iCARE. En esa submuestra la atribucion de
  la polaridad global a la entidad citada es mucho mas defendible, y sirve
  como analisis de sensibilidad frente al revisor.

Sobre que texto se aplica
-------------------------
Sobre `texto`, el ORIGINAL. Los modelos de sentimiento para Twitter se
entrenan con tuits crudos y dependen de la puntuacion, las mayusculas y
los emojis. Aplicarlos al texto lematizado degrada el resultado. Este es
el motivo por el que la etapa 02 conserva las tres representaciones.

Reanudacion: se procesa por bloques con checkpoint en
intermedios/s05_bloques/.

Salida
------
  intermedios/s05_polaridad.parquet
  intermedios/s05_corpus_analitico.parquet   <- corpus final del paper
  resultados/tablas/s05_distribucion_polaridad.csv
  resultados/tablas/s05_polaridad_por_semana.csv
  resultados/metricas/s05_polaridad.json
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

ETAPA = "s05"
DIR_BLOQUES = cfg.DIR_INTERMEDIOS / "s05_bloques"


def main() -> None:
    p = comun.parser_base(__doc__)
    p.add_argument("--cpu", action="store_true", help="Fuerza CPU (solo depuracion).")
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()
    DIR_BLOQUES.mkdir(parents=True, exist_ok=True)

    if comun.debe_saltar(cfg.PQ_ANALITICO, args.forzar, log):
        return
    if not cfg.PQ_ENTIDADES.exists():
        log.error("Falta %s. Ejecutar antes la etapa 04.", cfg.PQ_ENTIDADES.name)
        sys.exit(1)

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    df = pd.read_parquet(cfg.PQ_ENTIDADES)
    n = len(df)
    log.info("Corpus a clasificar: %s tuits", f"{n:,}")

    hay_gpu = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if hay_gpu else "cpu")
    log.info("Dispositivo: %s", device)
    if not hay_gpu:
        log.warning("Sin GPU esta etapa es muy lenta. Lanzar en la particion 'accel'.")

    tok = AutoTokenizer.from_pretrained(cfg.MODELO_POLARIDAD,
                                        revision=cfg.MODELO_POLARIDAD_REVISION)
    # [LOCAL] En una GPU de 6 GB los pesos se cargan directamente en fp16.
    # Reduce a la mitad la memoria del modelo y acelera la inferencia sin
    # efecto apreciable en las etiquetas, que se deciden por argmax.
    kw = {"revision": cfg.MODELO_POLARIDAD_REVISION}
    if hay_gpu and cfg.USAR_FP16:
        kw["torch_dtype"] = torch.float16
    modelo = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODELO_POLARIDAD, **kw).to(device).eval()
    if hay_gpu:
        libre, total = torch.cuda.mem_get_info()
        log.info("VRAM: %.1f GB libres de %.1f GB. Lote %d, longitud maxima %d.",
                 libre / 1e9, total / 1e9, cfg.POLARIDAD_BATCH, cfg.POLARIDAD_MAX_LEN)
        log.info("Si aparece 'CUDA out of memory', bajar POLARIDAD_BATCH en config.py.")

    # el orden de las etiquetas se lee del propio modelo, no se asume
    id2label = getattr(modelo.config, "id2label", None)
    log.info("Etiquetas declaradas por el modelo: %s", id2label)
    orden = [str(id2label[i]).capitalize() if id2label else cfg.POLARIDAD_ETIQUETAS[i]
             for i in range(modelo.config.num_labels)]
    orden = [{"Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive"}.get(o, o)
             for o in orden]
    log.info("Orden de clases usado: %s", orden)

    textos = df["texto"].fillna("").astype(str).values
    tam = cfg.CHUNK_TWEETS
    n_bloques = int(np.ceil(n / tam))
    log.info("Inferencia en %d bloques de %s (reanudable)", n_bloques, f"{tam:,}")

    @torch.no_grad()
    def inferir(lista) -> np.ndarray:
        salida = []
        for i in range(0, len(lista), cfg.POLARIDAD_BATCH):
            trozo = [t if t.strip() else "." for t in lista[i:i + cfg.POLARIDAD_BATCH]]
            enc = tok(trozo, truncation=True, padding=True,
                      max_length=cfg.POLARIDAD_MAX_LEN, return_tensors="pt").to(device)
            with comun.autocast(hay_gpu):
                logits = modelo(**enc).logits
            salida.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.vstack(salida) if salida else np.zeros((0, len(orden)))

    for b in range(n_bloques):
        destino = DIR_BLOQUES / f"bloque_{b:04d}.parquet"
        if destino.exists() and not args.forzar:
            log.info("  bloque %d/%d ya existe, se salta", b + 1, n_bloques)
            continue
        ini, fin = b * tam, min((b + 1) * tam, n)
        with comun.Cronometro(log, f"bloque {b + 1}/{n_bloques} (filas {ini:,}-{fin:,})"):
            probs = inferir(list(textos[ini:fin]))
            salida = pd.DataFrame(probs, columns=[f"prob_{o.lower()}" for o in orden])
            salida.insert(0, "idx", np.arange(ini, fin))
            salida.to_parquet(destino, index=False)

    piezas = [pd.read_parquet(f) for f in sorted(DIR_BLOQUES.glob("bloque_*.parquet"))]
    pol = pd.concat(piezas, ignore_index=True).sort_values("idx").reset_index(drop=True)
    if len(pol) != n:
        log.error("Los bloques suman %s filas y el corpus tiene %s. Borrar "
                  "intermedios/s05_bloques y relanzar con --forzar.", f"{len(pol):,}", f"{n:,}")
        sys.exit(1)

    cols = [f"prob_{o.lower()}" for o in orden]
    pol["polaridad"] = [orden[i] for i in pol[cols].values.argmax(axis=1)]
    pol["confianza_polaridad"] = pol[cols].values.max(axis=1)
    comun.guardar_parquet(pol, cfg.PQ_POLARIDAD, log)

    for c in cols + ["polaridad", "confianza_polaridad"]:
        df[c] = pol[c].values

    # ---------------- distribucion ----------------
    dist = (df["polaridad"].value_counts().rename_axis("polaridad")
            .reset_index(name="n"))
    dist["pct"] = (dist["n"] / n * 100).round(2)
    filas = [comun.fila_proporcion(r.polaridad, r.n, n) for r in dist.itertuples()]
    dist_ic = pd.DataFrame(filas).rename(columns={"categoria": "polaridad"})
    comun.guardar_tabla(dist_ic, "s05_distribucion_polaridad", log)
    log.info("Distribucion de polaridad:\n%s", dist_ic.to_string(index=False))

    log.info("Confianza media del clasificador: %.3f | tuits con confianza <0.5: %s (%.1f%%)",
             df["confianza_polaridad"].mean(),
             f"{int((df['confianza_polaridad'] < 0.5).sum()):,}",
             100 * (df["confianza_polaridad"] < 0.5).mean())

    # ---------------- serie temporal ----------------
    sem = (df.assign(semana=df["fecha"].dt.to_period("W").dt.start_time)
             .groupby(["semana", "polaridad"]).size().unstack(fill_value=0))
    sem_pct = (sem.div(sem.sum(axis=1), axis=0) * 100).round(2)
    sem_pct["n_tuits"] = sem.sum(axis=1)
    comun.guardar_tabla(sem_pct.reset_index(), "s05_polaridad_por_semana", log)

    comun.guardar_parquet(df, cfg.PQ_ANALITICO, log)
    comun.registrar_funnel(ETAPA, "E13_corpus_analitico", len(df),
                           "Corpus analitico final: origen colombiano, entidades y polaridad")

    comun.guardar_metricas({
        "modelo": cfg.MODELO_POLARIDAD,
        "revision": cfg.MODELO_POLARIDAD_REVISION,
        "id2label_del_modelo": {str(k): v for k, v in (id2label or {}).items()},
        "orden_de_clases_usado": orden,
        "texto_de_entrada": "texto original sin limpiar",
        "max_len": cfg.POLARIDAD_MAX_LEN, "batch": cfg.POLARIDAD_BATCH,
        "es_target_dependent": cfg.POLARIDAD_ES_TARGET_DEPENDENT,
        "advertencia_lenguaje": ("La polaridad es global del tuit. El manuscrito NO puede "
                                 "escribir 'sentiment toward an entity'; solo 'polarity of "
                                 "the discourse in which the entity is mentioned'."),
        "no_se_uso": ["pysentimiento", "RoBERTuito"],
        "n_tuits": n,
        "distribucion": dist_ic.to_dict("records"),
        "confianza_media": float(df["confianza_polaridad"].mean()),
        "pct_confianza_baja": round(100 * float((df["confianza_polaridad"] < 0.5).mean()), 2),
        "validacion_humana": "PENDIENTE — ver etapa 09.",
    }, "s05_polaridad", log)

    log.info("ETAPA 05 COMPLETADA. Corpus analitico: %s tuits", f"{len(df):,}")


if __name__ == "__main__":
    main()
