# -*- coding: utf-8 -*-
"""
comun.py — utilidades compartidas por todas las etapas del pipeline.

Tres cosas que resuelve:
  1. Logging uniforme a consola y a resultados/logs/<etapa>.log
  2. Registro del EMBUDO (funnel): cada etapa declara sus N y el
     acumulado se escribe en resultados/funnel/funnel.csv. Ese archivo
     alimenta directamente la Figura del embudo y la seccion 2.3.3 del
     manuscrito.
  3. Guardado con checkpoint: si un parquet ya existe y no se pidio
     --forzar, la etapa se salta. Asi un job caido se reanuda sin
     recomputar lo anterior.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def obtener_logger(etapa: str) -> logging.Logger:
    cfg.crear_directorios()
    log = logging.getLogger(etapa)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | " + etapa + " | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(cfg.DIR_LOGS / f"{etapa}.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info("=" * 70)
    log.info("Inicio de etapa %s", etapa)
    return log


# ---------------------------------------------------------------------------
# Embudo (funnel)
# ---------------------------------------------------------------------------
ARCHIVO_FUNNEL = None  # se resuelve en tiempo de ejecucion


def registrar_funnel(etapa: str, paso: str, n: int, descripcion: str = "") -> None:
    """Anade una fila al embudo. Se sobrescribe la fila si ya existia
    (misma etapa + mismo paso), para que re-ejecutar no duplique."""
    cfg.crear_directorios()
    ruta = cfg.DIR_FUNNEL / "funnel.csv"
    fila = {
        "etapa": etapa,
        "paso": paso,
        "n": int(n),
        "descripcion": descripcion,
        "momento": datetime.now().isoformat(timespec="seconds"),
    }
    if ruta.exists():
        df = pd.read_csv(ruta)
        df = df[~((df["etapa"] == etapa) & (df["paso"] == paso))]
        df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    else:
        df = pd.DataFrame([fila])
    df.to_csv(ruta, index=False)


def construir_funnel_ordenado(orden: list[tuple[str, str]]) -> pd.DataFrame:
    """Devuelve el embudo en el orden dado, con N y % retenido respecto
    del paso inicial y respecto del paso inmediatamente anterior."""
    ruta = cfg.DIR_FUNNEL / "funnel.csv"
    if not ruta.exists():
        return pd.DataFrame()
    df = pd.read_csv(ruta)
    filas = []
    for etapa, paso in orden:
        sub = df[(df["etapa"] == etapa) & (df["paso"] == paso)]
        if len(sub):
            filas.append(sub.iloc[-1])
    if not filas:
        return pd.DataFrame()
    out = pd.DataFrame(filas).reset_index(drop=True)
    n0 = out["n"].iloc[0]
    out["pct_del_inicial"] = (out["n"] / n0 * 100).round(2)

    # El "% del paso anterior" solo tiene sentido entre filtros sucesivos
    # sobre la MISMA unidad de analisis. Las evidencias paralelas (cada
    # una parte del corpus completo, no de la anterior) y los cambios de
    # unidad (de tuits a menciones) se marcan y se dejan en blanco, para
    # no sugerir una cadena que no existe.
    out["tipo_paso"] = [
        clasificar_paso(e, p) for e, p in zip(out["etapa"], out["paso"])]
    pct = []
    prev = None
    for i, r in out.iterrows():
        if r["tipo_paso"] == "secuencial" and prev is not None:
            pct.append(round(r["n"] / prev * 100, 2))
        else:
            pct.append(np.nan)
        if r["tipo_paso"] == "secuencial":
            prev = r["n"]
    out["pct_del_anterior"] = pct
    out.loc[0, "pct_del_anterior"] = 100.0
    return out[["etapa", "paso", "descripcion", "tipo_paso", "n",
                "pct_del_inicial", "pct_del_anterior"]]


# Pasos que NO son un filtro aplicado al resultado del paso anterior.
PASOS_EVIDENCIA_PARALELA = {"E08_seed_gazetteer", "E09_seed_grafo"}
PASOS_CAMBIO_DE_UNIDAD = {"E12_menciones_validas"}
# Subconjunto descriptivo del corpus, no un filtro que el paso siguiente
# herede: E13 se calcula sobre E10, no sobre E11.
PASOS_SUBCONJUNTO = {"E11_con_entidad"}


def clasificar_paso(etapa: str, paso: str) -> str:
    if paso in PASOS_EVIDENCIA_PARALELA:
        return "evidencia_paralela"
    if paso in PASOS_CAMBIO_DE_UNIDAD:
        return "cambio_de_unidad"
    if paso in PASOS_SUBCONJUNTO:
        return "subconjunto_descriptivo"
    return "secuencial"


# ---------------------------------------------------------------------------
# Guardado / checkpoint
# ---------------------------------------------------------------------------
def guardar_parquet(df: pd.DataFrame, ruta: Path, log=None) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, ruta)
    if log:
        mb = ruta.stat().st_size / 1e6
        log.info("Guardado %s  (%s filas, %s columnas, %.1f MB)",
                 ruta.name, f"{len(df):,}", df.shape[1], mb)


def guardar_tabla(df: pd.DataFrame, nombre: str, log=None, indice: bool = False) -> None:
    """Guarda una tabla de resultados en CSV. Es el formato que consumen
    tanto las figuras como el manuscrito."""
    cfg.crear_directorios()
    ruta = cfg.DIR_TABLAS / f"{nombre}.csv"
    df.to_csv(ruta, index=indice, encoding="utf-8")
    if log:
        log.info("Tabla guardada: %s (%s filas)", ruta.name, f"{len(df):,}")


def guardar_metricas(d: dict, nombre: str, log=None) -> None:
    cfg.crear_directorios()
    ruta = cfg.DIR_METRICAS / f"{nombre}.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(_json_seguro(d), f, indent=2, ensure_ascii=False)
    if log:
        log.info("Metricas guardadas: %s", ruta.name)


def _json_seguro(o):
    if isinstance(o, dict):
        return {str(k): _json_seguro(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_seguro(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return o


def debe_saltar(ruta: Path, forzar: bool, log=None) -> bool:
    if ruta.exists() and not forzar:
        if log:
            log.info("Ya existe %s y no se pidio --forzar. Se salta la etapa.", ruta.name)
            log.info("(Para recomputar: anadir --forzar)")
        return True
    return False


def parser_base(descripcion: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=descripcion)
    p.add_argument("--forzar", action="store_true",
                   help="Recomputa aunque ya exista el checkpoint de salida.")
    p.add_argument("--prueba", action="store_true",
                   help="Modo prueba: procesa solo una fraccion de los datos.")
    return p


class Cronometro:
    def __init__(self, log, etiqueta):
        self.log, self.etiqueta = log, etiqueta

    def __enter__(self):
        self.t0 = time.time()
        self.log.info("-> %s ...", self.etiqueta)
        return self

    def __exit__(self, *a):
        dt = time.time() - self.t0
        self.log.info("<- %s terminado en %s", self.etiqueta, formato_duracion(dt))


def formato_duracion(seg: float) -> str:
    h, r = divmod(int(seg), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Estadistica de apoyo
# ---------------------------------------------------------------------------
def ic_wilson(exitos: int, n: int, alfa: float = None) -> tuple[float, float, float]:
    """Intervalo de confianza de Wilson para una proporcion.

    Se usa Wilson y no Wald porque con n pequeno (los items de confianza
    tienen n~125) o proporciones cercanas a 0/1 el intervalo de Wald da
    limites fuera de [0,1] y cobertura pobre.

    Devuelve (proporcion, limite_inferior, limite_superior) en tanto por uno.
    """
    from scipy.stats import norm
    alfa = cfg.ALFA if alfa is None else alfa
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z = norm.ppf(1 - alfa / 2)
    p = exitos / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, centro - margen), min(1.0, centro + margen))


def fila_proporcion(etiqueta: str, exitos: int, n: int, decimales: int = 1) -> dict:
    """Fila estandar 'n/N (% [IC95%])' para las tablas del manuscrito."""
    p, lo, hi = ic_wilson(exitos, n)
    return {
        "categoria": etiqueta,
        "n": int(exitos),
        "valid_n": int(n),
        "pct": round(p * 100, decimales) if n else np.nan,
        "ic95_inf": round(lo * 100, decimales) if n else np.nan,
        "ic95_sup": round(hi * 100, decimales) if n else np.nan,
    }


def cramers_v(tabla: np.ndarray) -> dict:
    """Chi-cuadrado + V de Cramer con correccion de sesgo (Bergsma 2013).

    Con N grande el p-valor es casi siempre significativo, por eso el
    asesor pide explicitamente el tamano de efecto. Se devuelve tambien
    la V sin corregir para comparabilidad con la literatura.
    """
    from scipy.stats import chi2_contingency
    tabla = np.asarray(tabla, dtype=float)
    chi2, p, gl, esperados = chi2_contingency(tabla)
    n = tabla.sum()
    r, k = tabla.shape
    v = np.sqrt((chi2 / n) / (min(r, k) - 1))
    # correccion de sesgo
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(rc - 1, kc - 1)
    v_corr = np.sqrt(phi2c / denom) if denom > 0 else np.nan
    return {
        "chi2": float(chi2),
        "gl": int(gl),
        "p_valor": float(p),
        "n": int(n),
        "cramers_v": float(v),
        "cramers_v_corregida": float(v_corr),
        "interpretacion_v": interpretar_v(v, min(r, k) - 1),
        "frecuencia_esperada_minima": float(esperados.min()),
    }


def interpretar_v(v: float, gl_min: int) -> str:
    """Umbrales de Cohen adaptados a los grados de libertad."""
    if np.isnan(v):
        return "no calculable"
    base = 0.1 / np.sqrt(max(gl_min, 1))
    if v < base:
        return "insignificante"
    if v < 3 * base:
        return "pequeno"
    if v < 5 * base:
        return "mediano"
    return "grande"


def residuos_estandarizados(tabla: np.ndarray) -> np.ndarray:
    """Residuos estandarizados ajustados (Haberman). |r| > 1.96 senala la
    celda que aporta a la asociacion. Es el post-hoc que pide el asesor."""
    from scipy.stats import chi2_contingency
    tabla = np.asarray(tabla, dtype=float)
    _, _, _, esp = chi2_contingency(tabla)
    n = tabla.sum()
    fila = tabla.sum(axis=1, keepdims=True) / n
    col = tabla.sum(axis=0, keepdims=True) / n
    denom = np.sqrt(esp * (1 - fila) * (1 - col))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (tabla - esp) / denom


# ---------------------------------------------------------------------------
# Compatibilidad de precision mixta entre versiones de PyTorch
# ---------------------------------------------------------------------------
# torch.cuda.amp.autocast / GradScaler quedaron obsoletos en torch 2.4 a
# favor de torch.amp.autocast('cuda', ...) / torch.amp.GradScaler('cuda').
# APOLO puede tener cualquiera de las dos ramas segun cuando se instale el
# entorno, asi que se resuelve en tiempo de ejecucion en vez de fijar una.
def autocast(habilitado: bool):
    import torch
    try:
        return torch.amp.autocast("cuda", enabled=habilitado)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=habilitado)


def grad_scaler(habilitado: bool):
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=habilitado)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=habilitado)
