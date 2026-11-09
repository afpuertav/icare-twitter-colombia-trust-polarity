# -*- coding: utf-8 -*-
"""
ETAPA 08 — Todas las figuras del manuscrito, desde un unico script.

Que resuelve del informe del asesor
-----------------------------------
* Regla general: "Todas las figuras y tablas deben regenerarse desde la
  misma base analitica y script."

* Las figuras salen SIN titulo, sin numero de figura y sin notas al pie
  impresas: el pie de figura del manuscrito es el unico lugar donde se
  numera y se explica cada grafico.

  Este script NO recalcula nada. Lee exclusivamente los CSV que dejaron
  las etapas 06 y 07 en resultados/tablas/. Por construccion es imposible
  que una figura muestre un numero distinto del que aparece en el texto,
  que es el problema que el asesor detecto en las Figuras 2 y 5.

* #10 Figura 1: "Convertirla en arquitectura analitica con N por etapa,
  modelos usados y punto exacto de integracion." -> fig1
* #9  "Crear un diagrama tipo flowchart/funnel del corpus Twitter con N
  por etapa." -> fig2 (NUEVA; obliga a renumerar las siguientes)
* #14/#15 Figura de confianza: "Regenerar y utilizar una visualizacion
  mas simple (barra horizontal) con cifras identicas al texto." -> fig3
* #18 "Reemplazar el bubble chart por barras apiladas al 100% por
  entidad, conservando N totales en etiquetas." -> fig5
* #19/#20 Figura 5: "Redisenar como cross-source concordance y rotular
  claramente que son dos metricas, dos poblaciones y dos unidades de
  analisis diferentes." -> fig6

NUMERACION RESULTANTE (el embudo entra como Figura 2 y desplaza al resto)
  Figura 1  Arquitectura analitica con N por etapa
  Figura 2  Embudo del corpus de Twitter/X                 [NUEVA]
  Figura 3  Confianza y desconfianza declaradas, con IC95%
  Figura 4  Entidades iCARE citadas en Twitter/X
  Figura 5  Polaridad por entidad, barras apiladas al 100%
  Figura 6  Concordancia entre fuentes
  Figura 7  Polaridad en el tiempo                          [apoyo]

Sin dependencias de binarios externos: todo se dibuja con matplotlib, asi
que no hace falta que Graphviz este instalado en el nodo de computo.

Salida: resultados/figuras/*.png y *.pdf (300 dpi, listas para la revista)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Arc, Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle, Wedge)

import config as cfg
import comun

ETAPA = "s08"

# Paleta accesible: distinguible en escala de grises y para daltonismo.
C_NEG = "#C0392B"
C_NEU = "#B7B7B7"
C_POS = "#2471A3"
C_ICARE = "#7D3C98"
C_TW = "#1E8449"
C_CAJA = "#EAF2F8"
C_BORDE = "#2E4053"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


# Los registros del embudo se escriben en espanol, que es el idioma de
# trabajo del proyecto; las figuras van a una revista en ingles. Aqui se
# traduce solo lo que aparece impreso, sin tocar los logs ni los CSV.
ETIQUETAS_EMBUDO_EN = {
    "E00_declarado": "Messages declared across the 290 collection files",
    "E01_parseado": "Rows successfully parsed from the .tab files",
    "E02_con_fecha": "Rows with a valid timestamp",
    "E03_sin_retweets": "Retweets excluded",
    "E04_deduplicado": "Exact duplicates removed (overlap between the two daily collections)",
    "E05_espanol": "Restricted to Spanish-language messages",
    "E06_ventana": "Within the study window (31 Mar - 18 Aug 2020)",
    "E07_preprocesado": "Mentions and hashtags extracted before linguistic cleaning",
    "E10_corpus_colombia": "Classified as Colombian in origin (majority of 3 independent evidence sources)",
    "E11_con_entidad": "Messages citing at least one iCARE source category",
    "E12_menciones_validas": "Valid mentions resolved to an iCARE category (unit of analysis)",
    "E13_corpus_analitico": "Final analytical corpus (entity and polarity assigned)",
}

# Pasos que dibuja la Figura 1, en el mismo orden y con los mismos N que la
# Tabla 1 del manuscrito. Se declara explicitamente en vez de filtrar por
# tipo_paso porque la correccion 2.2 de la segunda ronda pide que la figura
# termine mostrando los dos ultimos pasos (14,739 mensajes que citan alguna
# categoria y 27,135 menciones validas) y que el cambio de unidad de analisis
# quede marcado. E13 no entra: repetiria el N de E10 y la Tabla 1 no lo tiene.
PASOS_EMBUDO = [
    "E00_declarado", "E01_parseado", "E02_con_fecha", "E03_sin_retweets",
    "E04_deduplicado", "E05_espanol", "E06_ventana", "E07_preprocesado",
    "E10_corpus_colombia", "E11_con_entidad", "E12_menciones_validas",
]

# A partir de este paso la unidad de analisis deja de ser el mensaje y pasa a
# ser la mencion: un mismo mensaje puede aportar mas de una mencion, asi que
# el porcentaje respecto del paso anterior no esta definido.
PASO_CAMBIO_DE_UNIDAD = "E12_menciones_validas"


def etiqueta_embudo(paso: str, descripcion: str) -> str:
    return ETIQUETAS_EMBUDO_EN.get(paso, str(descripcion))


# Numero con el que cada figura aparece EN EL MANUSCRITO. El pipeline genera
# alguna figura mas de las que el paper incluye (la distribucion de confianza
# de iCARE y la serie temporal son material suplementario), por eso el numero
# del archivo y el del paper no coinciden.
#
# IMPORTANTE: las figuras NO llevan titulo, numero de figura ni notas
# impresas. Todo eso vive en el pie de figura del manuscrito, que es la
# practica de las revistas: asi una renumeracion se resuelve editando el
# texto y nunca deja una imagen contradiciendo su propio pie. Este mapa se
# conserva solo para el log, que dice a que figura del paper corresponde
# cada archivo, y para nombrar los archivos al insertarlos en el Word.
FIGURA_EN_MANUSCRITO = {
    # Las figuras se numeran por ORDEN DE PRIMERA MENCION en el texto, que es
    # la regla del ICMJE y de la mayoria de revistas. El embudo del corpus se
    # cita antes que el esquema integrado, asi que el embudo es la Figura 1.
    "figura_2_embudo_corpus": "Figure 1",
    "figura_1_arquitectura": "Figure 2",
    "figura_4_entidades_citadas": "Figure 3",
    "figura_top_menciones": "Figure 4",
    "figura_5_polaridad_por_entidad": "Figure 5",
    "figura_6_concordancia_cross_source": "Figure 6",
    "figura_3_confianza_icare": "Supplementary Figure S1",
    "figura_7_polaridad_temporal": "Supplementary Figure S2",
}


def categorias_analizables(log):
    """Categorias cuya asignacion alcanza la precision minima para interpretarse.

    Sale de la etapa 09 (validacion humana). Si esa etapa no se ha corrido
    todavia, se devuelve None y las figuras se dibujan sin distinguir, con un
    aviso: es preferible una figura sin marcar a una figura que finja una
    validacion que no existe.
    """
    d = leer("s09_precision_mencion_por_categoria")
    if d is None or "analizable" not in d.columns:
        log.warning("No hay s09_precision_mencion_por_categoria.csv: las figuras "
                    "no distinguiran categorias fiables. Correr antes la etapa 09.")
        return None, {}
    ok = set(d.loc[d["analizable"].astype(bool), "categoria"])
    prec = dict(zip(d["categoria"], d["precision_mencion"]))
    return ok, prec


def guardar(fig, nombre: str, log) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(cfg.DIR_FIGURAS / f"{nombre}.{ext}")
    plt.close(fig)
    log.info("Figura guardada: %s.png / .pdf", nombre)


def leer(nombre: str):
    ruta = cfg.DIR_TABLAS / f"{nombre}.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


def caja(ax, x, y, w, h, texto, color=C_CAJA, tam=8, borde=C_BORDE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=color, ec=borde, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontsize=tam, zorder=3, linespacing=1.35)


def flecha(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 lw=1.0, color=C_BORDE, zorder=1))


# ---------------------------------------------------------------------------
# ICONOS VECTORIALES
# ---------------------------------------------------------------------------
# Se dibujan con primitivas de matplotlib, no con una fuente de simbolos ni
# con imagenes externas: asi el script produce exactamente la misma figura en
# cualquier maquina, que es lo que exige la regla de reproducibilidad del
# asesor. Cada icono se define dentro de un cuadrado centrado en el origen
# (coordenadas entre -1.2 y 1.2) y se escala con el parametro s.

BANDERA_CO = ("#FCD116", "#003893", "#CE1126")

# Contorno simplificado de Colombia (longitud, latitud). Es un icono, no un
# mapa: solo tiene que reconocerse de un vistazo a tamano pequeno.
CONTORNO_COLOMBIA = [
    (-71.1, 12.4), (-72.4, 11.7), (-73.4, 11.3), (-74.9, 11.1), (-75.6, 10.8),
    (-75.7, 9.4), (-76.9, 8.6), (-77.4, 8.5), (-77.2, 7.9), (-77.9, 7.2),
    (-77.4, 6.5), (-77.6, 5.5), (-77.3, 4.0), (-77.9, 2.6), (-78.6, 2.4),
    (-79.0, 1.6), (-78.9, 1.4), (-77.7, 0.8), (-77.5, 0.4), (-76.4, 0.4),
    (-75.3, -0.1), (-74.8, -0.2), (-74.4, -0.9), (-73.7, -1.2), (-72.9, -2.4),
    (-71.8, -2.3), (-70.9, -2.2), (-70.1, -2.6), (-70.0, -4.2), (-69.4, -1.1),
    (-69.4, 1.1), (-69.8, 1.1), (-67.3, 1.9), (-67.1, 2.8), (-67.9, 3.0),
    (-67.5, 3.8), (-67.9, 4.6), (-67.4, 6.2), (-68.0, 6.5), (-69.4, 6.1),
    (-70.1, 6.9), (-71.0, 6.9), (-72.0, 7.0), (-72.5, 7.5), (-72.4, 8.4),
    (-73.0, 9.2), (-72.9, 10.2), (-72.2, 11.2), (-71.3, 11.8),
]


def _medallon(ax, cx, cy, r, color):
    """Circulo blanco con borde de color donde se aloja cada icono."""
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=color, lw=1.5, zorder=4))


def ic_encuesta(ax, cx, cy, s, color):
    """Cuestionario en papel: la encuesta iCARE."""
    ax.add_patch(Rectangle((cx - 0.62 * s, cy - 1.0 * s), 1.24 * s, 2.0 * s,
                           fc="white", ec=color, lw=1.2, zorder=5))
    ax.add_patch(Rectangle((cx - 0.22 * s, cy + 0.86 * s), 0.44 * s, 0.26 * s,
                           fc=color, ec=color, lw=1.0, zorder=6))
    for k, dy in enumerate((0.45, 0.05, -0.35)):
        y = cy + dy * s
        ax.add_patch(Rectangle((cx - 0.44 * s, y - 0.10 * s), 0.20 * s, 0.20 * s,
                               fc=color if k < 2 else "white", ec=color,
                               lw=0.9, zorder=6))
        ax.plot([cx - 0.14 * s, cx + 0.46 * s], [y, y], color=color, lw=1.1, zorder=6)


def ic_personas(ax, cx, cy, s, color):
    """Personas: los participantes que respondieron los items de confianza."""
    for dx, esc in ((-0.62, 0.80), (0.62, 0.80), (0.0, 1.0)):
        x = cx + dx * s
        ax.add_patch(Wedge((x, cy - 0.55 * s), 0.52 * s * esc, 0, 180,
                           fc=color, ec="none", alpha=0.9, zorder=5))
        ax.add_patch(Circle((x, cy + 0.28 * s * esc), 0.28 * s * esc,
                            fc=color, ec="none", zorder=6))


def ic_binario(ax, cx, cy, s, color):
    """Binarizacion: cada respuesta se convierte en confia / no confia."""
    ax.add_patch(Rectangle((cx - 1.05 * s, cy - 0.45 * s), 0.90 * s, 0.90 * s,
                           fc="white", ec=color, lw=1.2, zorder=5))
    ax.plot([cx - 0.86 * s, cx - 0.66 * s, cx - 0.28 * s],
            [cx * 0 + cy - 0.02 * s, cy - 0.24 * s, cy + 0.30 * s],
            color=color, lw=1.8, zorder=6, solid_capstyle="round")
    ax.add_patch(Rectangle((cx + 0.15 * s, cy - 0.45 * s), 0.90 * s, 0.90 * s,
                           fc="white", ec=color, lw=1.2, zorder=5))
    for a, b in (((0.36, -0.24), (0.84, 0.30)), ((0.36, 0.30), (0.84, -0.24))):
        ax.plot([cx + a[0] * s, cx + b[0] * s], [cy + a[1] * s, cy + b[1] * s],
                color=color, lw=1.8, zorder=6, solid_capstyle="round")


def ic_barras(ax, cx, cy, s, color):
    """Barras con intervalo de confianza: el resultado de la encuesta."""
    for dx, h in ((-0.66, 0.75), (0.0, 1.25), (0.66, 0.95)):
        x = cx + dx * s
        ax.add_patch(Rectangle((x - 0.20 * s, cy - 0.95 * s), 0.40 * s, h * s,
                               fc=color, ec="none", alpha=0.9, zorder=5))
        yt = cy - 0.95 * s + h * s
        ax.plot([x, x], [yt - 0.14 * s, yt + 0.22 * s], color="#17202A",
                lw=1.1, zorder=6)
        ax.plot([x - 0.13 * s, x + 0.13 * s], [yt + 0.22 * s, yt + 0.22 * s],
                color="#17202A", lw=1.1, zorder=6)
    ax.plot([cx - 1.0 * s, cx + 1.0 * s], [cy - 0.95 * s, cy - 0.95 * s],
            color=color, lw=1.3, zorder=6)


def ic_burbuja(ax, cx, cy, s, color):
    """Burbuja de conversacion con almohadilla: la recoleccion en Twitter/X."""
    ax.add_patch(FancyBboxPatch((cx - 1.0 * s, cy - 0.28 * s), 2.0 * s, 1.25 * s,
                                boxstyle="round,pad=0,rounding_size=" + str(0.32 * s),
                                fc=color, ec=color, lw=1.0, alpha=0.9, zorder=5))
    ax.add_patch(Polygon([(cx - 0.38 * s, cy - 0.24 * s),
                          (cx - 0.16 * s, cy - 0.95 * s),
                          (cx + 0.06 * s, cy - 0.24 * s)],
                         closed=True, fc=color, ec="none", alpha=0.9, zorder=5))
    ax.text(cx, cy + 0.35 * s, "#", ha="center", va="center", color="white",
            fontsize=8.6 * s, fontweight="bold", zorder=6)


def ic_embudo(ax, cx, cy, s, color):
    """Embudo: la accion de filtrar (duplicados e idioma)."""
    ax.add_patch(Polygon([(cx - 0.95 * s, cy + 0.72 * s), (cx + 0.95 * s, cy + 0.72 * s),
                          (cx + 0.20 * s, cy - 0.10 * s), (cx + 0.20 * s, cy - 0.72 * s),
                          (cx - 0.20 * s, cy - 0.95 * s), (cx - 0.20 * s, cy - 0.10 * s)],
                         closed=True, fc=color, ec=color, lw=1.0, alpha=0.9, zorder=5))
    for dx in (-0.58, 0.0, 0.58):
        ax.add_patch(Circle((cx + dx * s, cy + 1.05 * s), 0.14 * s,
                            fc=color, ec="none", zorder=6))
    ax.add_patch(Circle((cx, cy - 1.18 * s), 0.12 * s, fc=color, ec="none", zorder=6))


def ic_colombia(ax, cx, cy, s, color):
    """Silueta de Colombia con los colores de la bandera: el filtro de origen."""
    pts = np.array(CONTORNO_COLOMBIA, dtype=float)
    lon, lat = pts[:, 0], pts[:, 1]
    esc = 2.15 * s / max(lon.max() - lon.min(), lat.max() - lat.min())
    x = cx + (lon - (lon.min() + lon.max()) / 2) * esc
    y = cy + (lat - (lat.min() + lat.max()) / 2) * esc
    contorno = np.column_stack([x, y])
    base = Polygon(contorno, closed=True, fc="white", ec="none", zorder=5)
    ax.add_patch(base)
    y0, alto = y.min(), y.max() - y.min()
    for desde, fraccion, col in ((0.50, 0.50, BANDERA_CO[0]),
                                 (0.25, 0.25, BANDERA_CO[1]),
                                 (0.00, 0.25, BANDERA_CO[2])):
        banda = Rectangle((x.min(), y0 + desde * alto), x.max() - x.min(),
                          fraccion * alto, fc=col, ec="none", alpha=0.92, zorder=6)
        ax.add_patch(banda)
        banda.set_clip_path(base)
    ax.add_patch(Polygon(contorno, closed=True, fc="none", ec=color, lw=1.1, zorder=7))


def ic_arroba(ax, cx, cy, s, color):
    """Arroba: la resolucion de las @menciones a categorias de fuente."""
    ax.text(cx, cy, "@", ha="center", va="center", color=color,
            fontsize=13.5 * s, fontweight="bold", zorder=6)


def ic_caritas(ax, cx, cy, s, color):
    """Tres caras: las tres clases de polaridad del clasificador."""
    for dx, col, forma in ((-0.72, C_NEG, "triste"), (0.0, C_NEU, "neutra"),
                           (0.72, C_POS, "feliz")):
        x = cx + dx * s
        ax.add_patch(Circle((x, cy), 0.34 * s, fc=col, ec="none", alpha=0.95, zorder=5))
        for ex in (-0.12, 0.12):
            ax.add_patch(Circle((x + ex * s, cy + 0.11 * s), 0.045 * s,
                                fc="white", ec="none", zorder=6))
        if forma == "neutra":
            ax.plot([x - 0.14 * s, x + 0.14 * s], [cy - 0.13 * s, cy - 0.13 * s],
                    color="white", lw=1.2, zorder=6, solid_capstyle="round")
        else:
            centro = cy - 0.30 * s if forma == "feliz" else cy - 0.02 * s
            t1, t2 = (200, 340) if forma == "feliz" else (20, 160)
            ax.add_patch(Arc((x, centro), 0.34 * s, 0.30 * s, theta1=t1, theta2=t2,
                             color="white", lw=1.2, zorder=6))


def ic_ranking(ax, cx, cy, s, color):
    """Dos escalas unidas rango a rango: la integracion es una concordancia
    de ordenamientos, no una comparacion de valores."""
    for i, dy in enumerate((0.66, 0.0, -0.66)):
        ax.add_patch(Circle((cx - 0.72 * s, cy + dy * s), 0.20 * s,
                            fc=C_ICARE, ec="none", zorder=6))
        ax.add_patch(Circle((cx + 0.72 * s, cy + dy * s), 0.20 * s,
                            fc=C_TW, ec="none", zorder=6))
        ax.plot([cx - 0.52 * s, cx + 0.52 * s], [cy + dy * s, cy + dy * s],
                color=color, lw=1.2, ls=(0, (2, 1.4)), zorder=5)


def tarjeta(ax, x, y, w, h, color, relleno, dibujar_icono, titulo, detalle, n_txt):
    """Una etapa del pipeline: medallon con icono, texto y el N que sobrevive."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=1.3",
                                fc=relleno, ec=color, lw=1.3, zorder=2))
    r = 0.36 * h
    _medallon(ax, x + 6.8, y + h / 2, r, color)
    dibujar_icono(ax, x + 6.8, y + h / 2, 0.58 * r, color)
    ax.text(x + 13.4, y + h * 0.63, titulo, fontsize=8.6, fontweight="bold",
            va="center", ha="left", color="#17202A", zorder=5)
    ax.text(x + 13.4, y + h * 0.27, detalle, fontsize=7.2,
            va="center", ha="left", color="#5D6D7E", zorder=5)
    # El N vive en su propia columna a la derecha, para que nunca se monte
    # sobre el titulo por larga que sea la cifra.
    if n_txt:
        ax.text(x + w - 9.0, y + h / 2, n_txt, fontsize=9.2, fontweight="bold",
                va="center", ha="center", color=color, zorder=5)


# ---------------------------------------------------------------------------
def fig1_arquitectura(log) -> None:
    """Figura 2 del manuscrito - arquitectura analitica en formato de diagrama
    con iconos: cada etapa lleva el simbolo de la accion que ejecuta
    (cuestionario, filtrado, origen colombiano, resolucion de menciones,
    polaridad) y el N que sobrevive a esa etapa."""
    fun = leer("s09_funnel_completo")
    ica = leer("s06_flujo_icare")

    def n_de(paso, defecto=""):
        if fun is None:
            return defecto
        f = fun[fun["paso"] == paso]
        return f"N = {int(f['n'].iloc[0]):,}" if len(f) else defecto

    def n_icare(i, defecto=""):
        if ica is None or len(ica) <= i:
            return defecto
        return f"N = {int(ica['n'].iloc[i]):,}"

    fig, ax = plt.subplots(figsize=(12, 7.6))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 76)
    ax.set_aspect("equal")
    ax.axis("off")

    XI, XD, W, H = 1.5, 63.5, 55.0, 8.6

    ax.text(XI + W / 2, 72.8, "SURVEY COMPONENT", ha="center", fontsize=10.5,
            fontweight="bold", color=C_ICARE)
    ax.text(XD + W / 2, 72.8, "DIGITAL DISCOURSE COMPONENT", ha="center",
            fontsize=10.5, fontweight="bold", color=C_TW)
    ax.plot([XI, XI + W], [71.3, 71.3], color=C_ICARE, lw=2.2, solid_capstyle="round")
    ax.plot([XD, XD + W], [71.3, 71.3], color=C_TW, lw=2.2, solid_capstyle="round")

    izq = [
        (ic_encuesta, "iCARE survey, Colombia",
         "waves 1-6  ·  self-administered online", n_icare(0)),
        (ic_personas, "Trust in information sources",
         "items trustinf_sq001-009  ·  waves 5-6  ·  Sep-Dec 2020", n_icare(2)),
        (ic_binario, "Binarization of each item",
         "{1,2} trust  ·  {3,4} distrust  ·  {5} missing", ""),
        (ic_barras, "Distrust % per category",
         "Wilson 95% confidence intervals", "8 categories"),
    ]
    der = [
        (ic_burbuja, "Twitter/X collection",
         "Spanish  ·  290 files  ·  31 Mar - 18 Aug 2020", n_de("E00_declarado")),
        (ic_embudo, "Cleaning and language filter",
         "exact duplicates removed  ·  Spanish only", n_de("E05_espanol")),
        (ic_colombia, "Colombian origin",
         "gazetteer  ·  5 ML models  ·  reply graph  ·  majority",
         n_de("E10_corpus_colombia")),
        (ic_arroba, "Resolution of @mentions",
         "curated dictionary + XLM-R zero-shot", n_de("E12_menciones_validas")),
        (ic_caritas, "Polarity of each message",
         "twitter-xlm-roberta-base-sentiment", "3 classes"),
    ]

    def columna(x, tarjetas, color, relleno, y_top, y_bot):
        n = len(tarjetas)
        hueco = ((y_top - y_bot) - n * H) / (n - 1)
        ys = [y_top - H - i * (H + hueco) for i in range(n)]
        for (icono, tit, det, ntx), y in zip(tarjetas, ys):
            tarjeta(ax, x, y, W, H, color, relleno, icono, tit, det, ntx)
        for i in range(n - 1):
            flecha(ax, (x + W / 2, ys[i]), (x + W / 2, ys[i + 1] + H))

    columna(XI, izq, C_ICARE, "#F6EFF9", 69.5, 16.5)
    columna(XD, der, C_TW, "#EAF7F1", 69.5, 16.5)

    ax.add_patch(FancyBboxPatch((20, 2.6), 76, 10.2,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                fc="#FDF2E0", ec="#B9770E", lw=1.6, zorder=2))
    _medallon(ax, 27.5, 7.7, 3.3, "#B9770E")
    ic_ranking(ax, 27.5, 7.7, 0.58 * 3.3, "#B9770E")
    ax.text(34.5, 9.4, "INTEGRATION - at the level of SOURCE CATEGORIES",
            fontsize=9.6, fontweight="bold", va="center", color="#7E5109", zorder=5)
    ax.text(34.5, 6.1, "Spearman rank concordance between stated distrust "
            "and negative mentions",
            fontsize=7.6, va="center", color="#7E5109", zorder=5)
    flecha(ax, (XI + W / 2, 16.5), (44.0, 12.9))
    flecha(ax, (XD + W / 2, 16.5), (76.0, 12.9))

    guardar(fig, "figura_1_arquitectura", log)

# ---------------------------------------------------------------------------
def fig2_embudo(log) -> None:
    """Figura 1 del manuscrito — embudo del corpus con N y % por etapa.

    Correccion 2.2 de la segunda ronda: la figura debe reproducir la misma
    secuencia de la Tabla 1, incluidos los dos ultimos pasos, y marcar que
    en el ultimo cambia la unidad de analisis. Antes se filtraba por
    tipo_paso == 'secuencial', lo que dejaba fuera E11 y E12 y terminaba en
    E13 con el mismo N que E10; el resultado era una figura que el pie
    describia como la secuencia completa sin serlo.
    """
    fun = leer("s09_funnel_completo")
    if fun is None:
        log.warning("Falta s09_funnel_completo.csv: se omite la Figura 1. "
                    "Ejecutar la etapa 07b.")
        return
    fun = fun.set_index("paso")
    faltan = [p for p in PASOS_EMBUDO if p not in fun.index]
    if faltan:
        log.warning("El embudo no tiene los pasos %s: se dibuja con los "
                    "disponibles.", ", ".join(faltan))
    orden = [p for p in PASOS_EMBUDO if p in fun.index]
    if not orden:
        log.warning("Embudo vacio: se omite la Figura 1.")
        return
    f = fun.loc[orden].reset_index()

    # Los porcentajes se recalculan sobre ESTA cadena y no se heredan del CSV:
    # el funnel completo incluye evidencias paralelas, de modo que su columna
    # pct_del_anterior no corresponde al paso anterior de la figura.
    n0 = f["n"].iloc[0]
    f["pct_ini"] = 100 * f["n"] / n0
    f["pct_prev"] = 100 * f["n"] / f["n"].shift(1)
    f.loc[f["paso"] == PASO_CAMBIO_DE_UNIDAD, "pct_prev"] = np.nan

    fig, ax = plt.subplots(figsize=(10.5, 0.92 * len(f) + 0.9))
    ax.set_xlim(0, 10); ax.set_ylim(-0.95, len(f)); ax.axis("off")

    for i, r in f.iterrows():
        y = len(f) - i - 1
        cambia = r["paso"] == PASO_CAMBIO_DE_UNIDAD
        ancho = max(1.1, 6.2 * r["n"] / n0)
        x = (6.4 - ancho) / 2 + 0.35
        color = C_TW if cambia else plt.cm.Blues(0.30 + 0.42 * (i / max(len(f) - 2, 1)))
        ax.add_patch(FancyBboxPatch((x, y + 0.16), ancho, 0.66,
                                    boxstyle="round,pad=0.008",
                                    fc=color, ec=C_BORDE, lw=1.0))
        ax.text(x + ancho / 2, y + 0.49, f"{int(r['n']):,}", ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color="white" if cambia else "#17202A")
        ax.text(7.25, y + 0.62, etiqueta_embudo(r["paso"], r["descripcion"])[:96],
                va="center", fontsize=8.3)
        pa = ("" if pd.isna(r["pct_prev"])
              else f"   |   {r['pct_prev']:.2f}% of previous step")
        unidad = "mentions" if cambia else "messages"
        ax.text(7.25, y + 0.33,
                f"{r['pct_ini']:.2f}% of initial{pa}   [unit: {unidad}]",
                va="center", fontsize=7.4, color="#566573")
        if i < len(f) - 1:
            flecha(ax, (3.55, y + 0.16), (3.55, y - 0.16))

    # linea que separa los pasos medidos en mensajes de los medidos en menciones
    if PASO_CAMBIO_DE_UNIDAD in list(f["paso"]):
        i_cambio = int(f.index[f["paso"] == PASO_CAMBIO_DE_UNIDAD][0])
        y_sep = len(f) - i_cambio - 1 + 0.98
        ax.plot([0.30, 9.85], [y_sep, y_sep], ls=(0, (4, 3)), lw=1.0,
                color="#7F8C8D", zorder=0)
        ax.text(0.30, -0.20,
                "Below the dashed line the unit of analysis changes from "
                "messages to mentions: a single message may contribute more "
                "than one mention,\nso no percentage of the previous step is "
                "defined for the final step.",
                fontsize=7.6, style="italic", color="#566573", va="top")

    guardar(fig, "figura_2_embudo_corpus", log)


# ---------------------------------------------------------------------------
def fig3_confianza(log) -> None:
    """Figura 3 — confianza/desconfianza por fuente, barras horizontales con IC95%."""
    tr = leer("s06_confianza_por_fuente")
    if tr is None:
        log.warning("Falta s06_confianza_por_fuente.csv: se omite la Figura 3.")
        return
    d = tr[tr["medida"] == "Distrust"].sort_values("pct").reset_index(drop=True)
    t = tr[tr["medida"] == "Trust"].set_index("clave")

    fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(d) + 1.2))
    y = np.arange(len(d))
    conf = [t.loc[c, "pct"] for c in d["clave"]]

    ax.barh(y, conf, color=C_POS, alpha=0.80, label="Trust (responses 1-2)", height=0.62)
    ax.barh(y, d["pct"], left=conf, color=C_NEG, alpha=0.85,
            label="Distrust (responses 3-4)", height=0.62)
    ax.errorbar(np.array(conf) + d["pct"] / 1, y, xerr=None, fmt="none")
    for i, r in d.iterrows():
        ax.plot([r["ic95_inf"] + 0, r["ic95_sup"]], [i - 0.40, i - 0.40],
                color="#17202A", lw=1.3, solid_capstyle="butt")
        ax.plot([r["ic95_inf"]], [i - 0.40], marker="|", color="#17202A", ms=6)
        ax.plot([r["ic95_sup"]], [i - 0.40], marker="|", color="#17202A", ms=6)
        ax.text(101.5, i, f"{r['pct']:.1f}%  (n={int(r['n'])}/{int(r['valid_n'])})",
                va="center", fontsize=7.8, color=C_NEG)

    ax.set_yticks(y)
    ax.set_yticklabels(d["categoria"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of respondents (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    guardar(fig, "figura_3_confianza_icare", log)


# ---------------------------------------------------------------------------
def fig4_entidades_citadas(log) -> None:
    """Figura 4 — volumen de menciones por categoria iCARE en Twitter/X."""
    t = leer("s07_tabla3_entidad_polaridad_menciones")
    if t is None:
        log.warning("Falta s07_tabla3...csv: se omite la Figura 4.")
        return
    ok, prec = categorias_analizables(log)
    d = t.sort_values("n_total")
    total = d["n_total"].sum()

    fig, ax = plt.subplots(figsize=(9.6, 0.52 * len(d) + 1.2))
    y = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        fiable = ok is None or r.categoria in ok
        ax.barh(i, r.n_total, height=0.62,
                color=C_TW if fiable else "#FFFFFF",
                edgecolor=C_TW, lw=1.1, alpha=0.85 if fiable else 1.0,
                hatch=None if fiable else "///")
        etq = f"{int(r.n_total):,}  ({100 * r.n_total / total:.1f}%)"
        if prec.get(r.categoria) is not None:
            etq += f"   precision {prec[r.categoria]:.2f}"
        ax.text(r.n_total * 1.02, i, etq, va="center", fontsize=7.8,
                color="#17202A" if fiable else "#7B7D7D")
    ax.set_yticks(y); ax.set_yticklabels(d["etiqueta_en"])
    ax.set_xlabel("Number of valid mentions")
    ax.set_xlim(0, d["n_total"].max() * 1.42)
    if ok is not None:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=C_TW, alpha=0.85, label="Interpreted in the analysis"),
                           Patch(facecolor="white", edgecolor=C_TW, hatch="///",
                                 label="Precision below threshold - not interpreted")],
                  loc="lower right", frameon=False, fontsize=7.6)
    guardar(fig, "figura_4_entidades_citadas", log)


# ---------------------------------------------------------------------------
def fig_top_menciones(log) -> None:
    """Figura 4 del manuscrito - las cuentas mas citadas del corpus colombiano."""
    d = leer("s04_top_menciones_corpus")
    if d is None:
        log.warning("Falta s04_top_menciones_corpus.csv: se omite la figura de "
                    "cuentas mas mencionadas. Correr la etapa 04 (--solo-tablas basta).")
        return
    d = d.sort_values("n_menciones")
    fig, ax = plt.subplots(figsize=(9.2, 0.46 * len(d) + 1.1))
    curado = d["origen"].eq("R1_diccionario_curado")
    ax.barh(np.arange(len(d)), d["n_menciones"], height=0.6,
            color=[C_TW if c else "#7FB3A3" for c in curado], alpha=0.9)
    for i, r in enumerate(d.itertuples()):
        ax.text(r.n_menciones * 1.02, i, f"{int(r.n_menciones):,}  ({r.pct_del_total:.1f}%)",
                va="center", fontsize=7.8)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([f"{h}\n{cfg.ETIQUETA_EN.get(c, c)}" for h, c
                        in zip(d["handle"], d["categoria"])], fontsize=7.6)
    ax.set_xlabel("Number of valid mentions")
    ax.set_xlim(0, d["n_menciones"].max() * 1.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_TW, alpha=0.9, label="Category from the curated dictionary"),
                       Patch(facecolor="#7FB3A3", alpha=0.9, label="Category from zero-shot classification")],
              loc="lower right", frameon=False, fontsize=7.6)
    guardar(fig, "figura_top_menciones", log)


# ---------------------------------------------------------------------------
def fig5_polaridad_apilada(log) -> None:
    """Figura 5 — barras apiladas al 100% por entidad (sustituye el bubble chart)."""
    t = leer("s07_tabla3_entidad_polaridad_menciones")
    if t is None:
        log.warning("Falta s07_tabla3...csv: se omite la Figura 5.")
        return
    ok, _ = categorias_analizables(log)
    if ok is not None:
        t = t[t["categoria"].isin(ok)]
    d = t.sort_values("pct_negative").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(d) + 1.3))
    y = np.arange(len(d))
    izq = np.zeros(len(d))
    for col, color, etiqueta in [("pct_negative", C_NEG, "Negative"),
                                 ("pct_neutral", C_NEU, "Neutral"),
                                 ("pct_positive", C_POS, "Positive")]:
        if col not in d.columns:
            continue
        ax.barh(y, d[col], left=izq, color=color, height=0.64, label=etiqueta,
                edgecolor="white", lw=0.6)
        for i, v in enumerate(d[col]):
            if v >= 6:
                ax.text(izq[i] + v / 2, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=7.4, color="white", fontweight="bold")
        izq = izq + d[col].values

    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}\n(N = {int(n):,})" for e, n
                        in zip(d["etiqueta_en"], d["n_total"])])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of mentions (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    guardar(fig, "figura_5_polaridad_por_entidad", log)


# ---------------------------------------------------------------------------
def fig6_concordancia(log) -> None:
    """Figura 6 — concordancia entre fuentes, rediseñada como dos escalas
    separadas unidas por rango. Deliberadamente NO se dibujan las dos
    metricas sobre un mismo eje, que es lo que hacia la figura anterior
    sugerir equivalencia directa."""
    c = leer("s07_concordancia_cross_source")
    if c is None:
        log.warning("Falta s07_concordancia_cross_source.csv: se omite la Figura 6.")
        return
    ok, _ = categorias_analizables(log)
    c = c.sort_values("icare_pct_distrust", ascending=False).reset_index(drop=True)
    c["fiable"] = True if ok is None else c["clave"].isin(ok)
    n = len(c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 0.62 * n + 1.4),
                                   gridspec_kw={"wspace": 0.55})
    y = np.arange(n)[::-1]

    # El panel de iCARE va siempre solido: el item de confianza de la encuesta
    # esta bien medido. Lo que puede fallar es identificar la categoria EN
    # TWITTER, asi que el rayado corresponde al panel de la derecha.
    ax1.barh(y, c["icare_pct_distrust"], color=C_ICARE, alpha=0.85, height=0.6)
    for i, r in c.iterrows():
        ax1.plot([r["icare_ic95_inf"], r["icare_ic95_sup"]], [y[i], y[i]],
                 color="#17202A", lw=1.2)
        for x in (r["icare_ic95_inf"], r["icare_ic95_sup"]):
            ax1.plot([x], [y[i]], marker="|", color="#17202A", ms=5)
        # la etiqueta va DESPUES del limite superior del IC, no pegada a la
        # barra, para que no se solape con el intervalo
        ax1.text(r["icare_ic95_sup"] + 2, y[i],
                 f"{r['icare_pct_distrust']:.1f}%   #{int(r['rango_icare'])}",
                 va="center", fontsize=7.6)
    ax1.set_yticks(y); ax1.set_yticklabels(c["etiqueta_en"], fontsize=8)
    ax1.set_xlabel("% of respondents reporting distrust")
    ax1.set_title("(A)", fontsize=10, loc="left", fontweight="bold")
    ax1.set_xlim(0, max(115, c["icare_ic95_sup"].max() * 1.30))
    ax2.barh(y, c["twitter_pct_negative"], height=0.6,
             color=[C_TW if f else "#FFFFFF" for f in c["fiable"]],
             edgecolor=C_TW, lw=1.1,
             hatch=[None if f else "///" for f in c["fiable"]], alpha=0.85)
    for i, r in c.iterrows():
        ax2.plot([r["twitter_ic95_inf"], r["twitter_ic95_sup"]], [y[i], y[i]],
                 color="#17202A", lw=1.2)
        for x in (r["twitter_ic95_inf"], r["twitter_ic95_sup"]):
            ax2.plot([x], [y[i]], marker="|", color="#17202A", ms=5)
        ax2.text(r["twitter_ic95_sup"] + 2, y[i],
                 f"{r['twitter_pct_negative']:.1f}%   #{int(r['rango_twitter'])}"
                 f"   (N={int(r['twitter_n_menciones']):,})",
                 va="center", fontsize=7.2)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel("% of mentions classified as negative")
    ax2.set_title("(B)", fontsize=10, loc="left", fontweight="bold")
    ax2.set_xlim(0, max(130, c["twitter_ic95_sup"].max() * 1.35))


    guardar(fig, "figura_6_concordancia_cross_source", log)


# ---------------------------------------------------------------------------
def fig7_serie_temporal(log) -> None:
    """Figura 7 (apoyo) — polaridad semanal."""
    s = leer("s05_polaridad_por_semana")
    if s is None:
        log.warning("Falta s05_polaridad_por_semana.csv: se omite la Figura 7.")
        return
    s["semana"] = pd.to_datetime(s["semana"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for col, color in [("Negative", C_NEG), ("Neutral", C_NEU), ("Positive", C_POS)]:
        if col in s.columns:
            ax.plot(s["semana"], s[col], marker="o", ms=3.4, color=color, label=col, lw=1.6)
    ax.set_ylabel("Percentage of tweets (%)")
    ax.set_xlabel("Week")
    ax.legend(frameon=False, ncol=3)
    fig.autofmt_xdate()
    guardar(fig, "figura_7_polaridad_temporal", log)


# ---------------------------------------------------------------------------
def main() -> None:
    p = comun.parser_base(__doc__)
    p.parse_args()
    log = comun.obtener_logger(ETAPA)
    cfg.crear_directorios()

    log.info("Generando figuras. Fuente unica: los CSV de resultados/tablas/.")
    for f in (fig1_arquitectura, fig2_embudo, fig3_confianza, fig4_entidades_citadas,
              fig_top_menciones, fig5_polaridad_apilada, fig6_concordancia,
              fig7_serie_temporal):
        try:
            f(log)
        except Exception as e:  # noqa: BLE001
            log.error("Fallo %s: %s: %s", f.__name__, type(e).__name__, e)
            import traceback
            log.error(traceback.format_exc())

    log.info("Correspondencia archivo -> numero en el manuscrito:")
    for arch, num in FIGURA_EN_MANUSCRITO.items():
        log.info("   %-38s %s", arch + ".png", num)
    log.info("ETAPA 08 COMPLETADA. Figuras en %s", cfg.DIR_FIGURAS)


if __name__ == "__main__":
    main()
