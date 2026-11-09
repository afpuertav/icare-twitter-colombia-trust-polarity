# -*- coding: utf-8 -*-
"""
ETAPA 03b (parte 2 de 2) — Origen colombiano, quinto modelo: BETO.

Que resuelve del informe del asesor
-----------------------------------
* Correccion #7: el manuscrito habla de "cinco modelos ML". Los cuatro
  clasicos estan en s03b_modelos_clasicos.py; este es el quinto.

Modelo
------
  dccuchile/bert-base-spanish-wwm-cased ("BETO"), BERT preentrenado en
  espanol con whole-word masking, ajustado (fine-tuning) para
  clasificacion binaria de origen colombiano.

  Se elige BETO y no un BERT multilingue porque el corpus es
  monolingue espanol y BETO rinde mejor en tareas en espanol con el mismo
  presupuesto de computo.

Diferencia importante con los modelos clasicos
----------------------------------------------
  Los clasicos usan `texto_limpio` (lematizado, sin stopwords). BETO usa
  el TEXTO ORIGINAL: los transformers preentrenados esperan lenguaje
  natural completo y pierden rendimiento si se les quita la morfologia y
  las palabras funcionales. Esto NO rompe la comparabilidad de la
  evaluacion, porque la particion train/val/test es la misma (las mismas
  filas, misma semilla, cargadas desde s03b_particion_geo.parquet); solo
  cambia la representacion de entrada, que es precisamente lo que se
  compara entre familias de modelos.

Requiere GPU. Lanzar en la particion `accel` con --gres=gpu:1.

Salida
------
  intermedios/s03b_seed_transformer.parquet
  resultados/tablas/s03b_metricas_transformer.csv
  resultados/figuras/s03b_matriz_confusion_beto.png
  resultados/metricas/s03b_transformer.json
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
import s03_datos_geo as datos

ETAPA = "s03b_beto"
NOMBRE = "beto"


def main() -> None:
    p = comun.parser_base(__doc__)
    p.add_argument("--cpu", action="store_true",
                   help="Fuerza CPU. Solo para depurar: en CPU esta etapa es inviable.")
    args = p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    if comun.debe_saltar(cfg.PQ_SEED2B, args.forzar, log):
        return

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)

    hay_gpu = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if hay_gpu else "cpu")
    log.info("Dispositivo: %s", device)
    if hay_gpu:
        libre, total = torch.cuda.mem_get_info()
        log.info("VRAM: %.1f GB libres de %.1f GB. Lote de entrenamiento %d, "
                 "de inferencia %d, longitud maxima %d.",
                 libre / 1e9, total / 1e9, cfg.GEO_BERT_BATCH,
                 cfg.GEO_BERT_BATCH_INFERENCIA, cfg.GEO_BERT_MAX_LEN)
        log.info("Si aparece 'CUDA out of memory', bajar GEO_BERT_BATCH en config.py.")
    if not hay_gpu:
        log.warning("Sin GPU. Esta etapa tardara dias sobre 1.4M de tuits. "
                    "Lanzar en la particion 'accel' con --gres=gpu:1.")

    split = datos.construir_particion(log, forzar=False)
    # BETO usa el texto original: hay que recuperarlo del parquet de la etapa 02
    textos = pd.read_parquet(cfg.PQ_TWITTER_PREP, columns=["texto"])["texto"].fillna("").values
    split = split.copy()
    split["texto_original"] = textos[split["idx"].values]

    tr = datos.balancear_entrenamiento(split[split["particion"] == "train"], log)
    va = split[split["particion"] == "val"]
    te = split[split["particion"] == "test"]

    tok = AutoTokenizer.from_pretrained(cfg.MODELO_GEO_BERT)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODELO_GEO_BERT, num_labels=2).to(device)

    class DS(Dataset):
        def __init__(self, textos, etiquetas=None):
            self.t = list(textos)
            self.y = None if etiquetas is None else np.asarray(etiquetas, dtype=np.int64)

        def __len__(self):
            return len(self.t)

        def __getitem__(self, i):
            return (self.t[i], -1 if self.y is None else int(self.y[i]))

    def colacion(lote):
        txt = [x[0] for x in lote]
        y = torch.tensor([x[1] for x in lote], dtype=torch.long)
        enc = tok(txt, truncation=True, padding=True,
                  max_length=cfg.GEO_BERT_MAX_LEN, return_tensors="pt")
        return enc, y

    dl_tr = DataLoader(DS(tr["texto_original"], tr["etiqueta"]),
                       batch_size=cfg.GEO_BERT_BATCH, shuffle=True, collate_fn=colacion)

    opt = torch.optim.AdamW(modelo.parameters(), lr=2e-5, weight_decay=0.01)
    pasos = len(dl_tr) * cfg.GEO_BERT_EPOCAS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * pasos), pasos)
    escala = comun.grad_scaler(hay_gpu)

    log.info("Entrenando BETO: %d epocas, %d pasos, batch %d, lr 2e-5, max_len %d",
             cfg.GEO_BERT_EPOCAS, pasos, cfg.GEO_BERT_BATCH, cfg.GEO_BERT_MAX_LEN)
    modelo.train()
    for ep in range(cfg.GEO_BERT_EPOCAS):
        with comun.Cronometro(log, f"epoca {ep + 1}/{cfg.GEO_BERT_EPOCAS}"):
            acumulada = 0.0
            for i, (enc, y) in enumerate(dl_tr, 1):
                enc = {k: v.to(device) for k, v in enc.items()}
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                with comun.autocast(hay_gpu):
                    out = modelo(**enc, labels=y)
                escala.scale(out.loss).backward()
                escala.step(opt)
                escala.update()
                sched.step()
                acumulada += out.loss.item()
                if i % 200 == 0:
                    log.info("   paso %d/%d  perdida media %.4f", i, len(dl_tr), acumulada / i)

    @torch.no_grad()
    def predecir(textos_lista) -> np.ndarray:
        modelo.eval()
        # [LOCAL] lote de inferencia propio: sin gradientes cabe mas que
        # en entrenamiento, pero no cuatro veces mas en una GPU de 6 GB.
        dl = DataLoader(DS(textos_lista), batch_size=cfg.GEO_BERT_BATCH_INFERENCIA,
                        shuffle=False, collate_fn=colacion)
        out = []
        for enc, _ in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            with comun.autocast(hay_gpu):
                logits = modelo(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    with comun.Cronometro(log, "evaluando en validacion y prueba"):
        prob_va = predecir(va["texto_original"].tolist())
        prob_te = predecir(te["texto_original"].tolist())

    y_va = va["etiqueta"].astype(int).values
    y_te = te["etiqueta"].astype(int).values
    u_opt, f1_val = datos.umbral_optimo_f1(y_va, prob_va, log)
    u = u_opt if cfg.GEO_OPTIMIZAR_UMBRAL else cfg.GEO_UMBRAL_FIJO
    log.info("Umbral optimo en validacion: %.3f (F1=%.3f)", u_opt, f1_val)

    m_opt = datos.evaluar(y_te, prob_te, (prob_te >= u).astype(int), NOMBRE)
    m_opt.update({"umbral": round(u, 4), "tipo_umbral": "optimo_f1_validacion"})
    m_fijo = datos.evaluar(y_te, prob_te, (prob_te >= cfg.GEO_UMBRAL_FIJO).astype(int), NOMBRE)
    m_fijo.update({"umbral": cfg.GEO_UMBRAL_FIJO, "tipo_umbral": "fijo_heredado_v1"})

    met = pd.DataFrame([m_opt, m_fijo]).round(4)
    comun.guardar_tabla(met, "s03b_metricas_transformer", log)
    log.info("\n%s", met.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_te, (prob_te >= u).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Other", "Colombia"]).plot(
        cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix — BETO\n(test set, natural class balance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(cfg.DIR_FIGURAS / "s03b_matriz_confusion_beto.png", dpi=200)
    plt.close(fig)

    with comun.Cronometro(log, "inferencia de BETO sobre el corpus completo"):
        prob_corpus = predecir(split["texto_original"].tolist())

    salida = pd.DataFrame({
        "idx": split["idx"].values,
        f"prob_{NOMBRE}": prob_corpus.astype(np.float32),
        f"pred_{NOMBRE}": (prob_corpus >= u).astype(np.int8),
    })
    comun.guardar_parquet(salida, cfg.PQ_SEED2B, log)

    comun.guardar_metricas({
        "modelo": cfg.MODELO_GEO_BERT,
        "entrada": "texto original (no lematizado)",
        "justificacion_entrada": ("los transformers preentrenados se degradan con texto "
                                  "lematizado y sin stopwords; la particion es identica "
                                  "a la de los modelos clasicos"),
        "epocas": cfg.GEO_BERT_EPOCAS, "batch": cfg.GEO_BERT_BATCH,
        "max_len": cfg.GEO_BERT_MAX_LEN, "lr": 2e-5, "optimizador": "AdamW",
        "scheduler": "linear con 10% warmup", "precision_mixta": hay_gpu,
        "n_train_balanceado": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "umbral_optimo_validacion": u_opt, "f1_validacion": f1_val, "umbral_aplicado": u,
        "metricas_test": m_opt, "metricas_test_umbral_fijo": m_fijo,
    }, "s03b_transformer", log)

    log.info("ETAPA 03b (BETO) COMPLETADA.")


if __name__ == "__main__":
    main()
