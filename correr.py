# -*- coding: utf-8 -*-
"""
correr.py — lanzador del pipeline en el equipo local, una etapa a la vez.

Sustituye a los .sbatch de la version de APOLO. Aqui no hay planificador:
cada etapa se ejecuta en el momento, en este mismo proceso.

USO
    python correr.py estado          que esta hecho y que se puede correr
    python correr.py 01              ejecuta la etapa 01
    python correr.py siguiente       ejecuta la primera pendiente
    python correr.py todo            encadena todas las pendientes
    python correr.py 01 --forzar     recomputa aunque ya tenga checkpoint

Cualquier otra opcion (--prueba, --sin-lematizar, --cpu) se pasa tal cual
al script de la etapa.

Antes de ejecutar comprueba las dependencias: si a una etapa le falta el
insumo de otra, lo dice y no la lanza. Y si la etapa ya tiene su resultado
en disco no la repite salvo --forzar.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config as cfg  # noqa: E402


# id, script, checkpoint que produce, dependencias, descripcion, usa GPU
ETAPAS = [
    ("01",   "s01_cargar_twitter.py",      cfg.PQ_TWITTER_CRUDO,  [],
     "Carga los .tab, deduplica, cobertura temporal", False),
    ("02",   "s02_preprocesar.py",         cfg.PQ_TWITTER_PREP,   ["01"],
     "Menciones y hashtags, luego lematizacion", False),
    ("03a",  "s03a_gazetteer.py",          cfg.PQ_SEED1,          ["02"],
     "Evidencia 1: gazetteer + lexicon + Location", False),
    ("03b",  "s03b_modelos_clasicos.py",   cfg.PQ_SEED2,          ["02"],
     "Evidencia 2a: NB, SVM, RF y XGBoost", False),
    ("03bt", "s03b_modelo_transformer.py", cfg.PQ_SEED2B,         ["02"],
     "Evidencia 2b: BETO", True),
    ("03c",  "s03c_grafo.py",              cfg.PQ_SEED3,          ["03a", "03b"],
     "Evidencia 3: grafo de respuestas", False),
    ("03d",  "s03d_corpus_colombia.py",    cfg.PQ_COLOMBIA,       ["03a", "03b", "03c"],
     "Regla de combinacion y corpus colombiano", False),
    ("04",   "s04_entidades.py",           cfg.PQ_ENTIDADES,      ["03d"],
     "Entidades: diccionario, reglas y zero-shot", True),
    ("05",   "s05_polaridad.py",           cfg.PQ_ANALITICO,      ["04"],
     "Polaridad con twitter-xlm-roberta", True),
    ("06",   "s06_icare.py",               cfg.PQ_ICARE,          [],
     "iCARE: flujo, Tabla 1 y confianza", False),
    ("07",   "s07_estadistica.py",
     cfg.DIR_TABLAS / "s07_tabla3_entidad_polaridad_menciones.csv", ["05", "06"],
     "chi2, V de Cramer, IC95%, concordancia", False),
    ("07b",  "s07b_consolidar_validar.py",
     cfg.DIR_TABLAS / "s09_funnel_completo.csv", ["07"],
     "Embudo, muestras de validacion, auditoria", False),
    ("09",   "s09_evaluar_anotacion.py",
     cfg.DIR_METRICAS / "s09_validacion_humana.json", ["07b"],
     "Validacion humana: kappa y precision (antes que las figuras)", False),
    ("08",   "s08_figuras.py",
     cfg.DIR_FIGURAS / "figura_1_arquitectura.png", ["07b"],
     "Todas las figuras (usa la validacion de la 09 si existe)", False),
    ("10",   "s10_sensibilidad.py",
     cfg.DIR_TABLAS / "s10_tablaA2_comparacion_corpus.csv", ["07", "09"],
     "Sensibilidad: corpus geografico unanime y diccionario curado", False),
]

POR_ID = {e[0]: e for e in ETAPAS}


def hecha(id_: str) -> bool:
    return Path(POR_ID[id_][2]).exists()


def formato_duracion(seg: float) -> str:
    h, r = divmod(int(seg), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


def estado() -> str | None:
    siguiente = None
    print()
    print(f"{'ETAPA':<6}{'ESTADO':<9}{'GPU':<5}{'DESCRIPCION':<48}PRODUCE")
    print("-" * 104)
    for id_, script, salida, deps, desc, gpu in ETAPAS:
        if hecha(id_):
            est = "HECHA"
        elif any(not hecha(d) for d in deps):
            est = "espera"
        else:
            est = "LISTA"
            if siguiente is None and id_ != "09":
                siguiente = id_
        print(f"{id_:<6}{est:<9}{'si' if gpu else '':<5}{desc:<48}{Path(salida).name}")
    print("-" * 104)
    print("HECHA = ya tiene su resultado    LISTA = se puede correr ahora")
    print("espera = le falta el insumo de otra etapa")
    print()
    print(f"Directorio de trabajo: {cfg.DIR_TRABAJO}")
    print(f"Datos de entrada:      {cfg.DIR_DATOS}")
    if siguiente:
        print(f"\nSiguiente:  python correr.py {siguiente}")
    else:
        print("\nNo hay etapas pendientes que puedan correr ahora.")
    print()
    return siguiente


# Paquetes que necesita cada etapa. Se comprueban ANTES de lanzarla: la 04
# descarga 2.2 GB de modelo antes de tocar el tokenizador, y descubrir ahi
# que falta protobuf cuesta tres minutos y un error que no dice de que va.
BASE = ("pandas", "numpy", "pyarrow", "scipy", "openpyxl")
POR_ETAPA = {
    "01": BASE,
    "02": BASE + ("spacy",),
    "03a": BASE,
    "03b": BASE + ("sklearn", "xgboost", "matplotlib"),
    "03bt": BASE + ("sklearn", "torch", "transformers"),
    "03c": BASE,
    "03d": BASE,
    # sentencepiece y protobuf: el tokenizador de XLM-RoBERTa es
    # SentencePiece y transformers lo convierte al cargarlo.
    "04": BASE + ("torch", "transformers", "sentencepiece", "google.protobuf"),
    "05": BASE + ("torch", "transformers", "sentencepiece", "google.protobuf"),
    "06": BASE,
    "07": BASE,
    "07b": BASE,
    "08": BASE + ("matplotlib",),
    "09": BASE + ("sklearn",),
}

# como se instala cada uno, para poder decirlo en el mensaje de error
COMO_INSTALAR = {
    "sentencepiece": "correr.bat reparar-tokenizer",
    "google.protobuf": "correr.bat reparar-tokenizer",
    "torch": "correr.bat reparar-gpu",
    "spacy": "correr.bat reparar-spacy",
}


def comprobar_entorno(id_: str | None = None) -> bool:
    """Avisa de lo que falta antes de que falle a mitad de una etapa."""
    ok = True
    requeridos = POR_ETAPA.get(id_, BASE) if id_ else BASE
    faltan = [m for m in requeridos if not _hay(m)]
    if faltan:
        print(f"Faltan paquetes para la etapa {id_ or '?'}: {', '.join(faltan)}")
        remedios = {COMO_INSTALAR.get(m, "conda env create -f entorno.yml") for m in faltan}
        for r in sorted(remedios):
            print(f"  {r}")
        ok = False
    if not (cfg.DIR_DATOS / "dfTwitter").is_dir():
        print(f"No se encuentra la carpeta de datos: {cfg.DIR_DATOS / 'dfTwitter'}")
        ok = False
    return ok


def _hay(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def ejecutar(id_: str, extra: list[str]) -> int:
    if id_ not in POR_ID:
        print(f"Etapa desconocida: {id_}")
        print(f"Validas: {', '.join(e[0] for e in ETAPAS)}")
        return 1
    _, script, salida, deps, desc, gpu = POR_ID[id_]

    if not comprobar_entorno(id_):
        return 1

    falta = [d for d in deps if not hecha(d)]
    if falta:
        print(f"ERROR: a la etapa {id_} le falta el resultado de: {' '.join(falta)}\n")
        for d in falta:
            print(f"  python correr.py {d}      ({POR_ID[d][4]})")
        print("\nPanorama completo:  python correr.py estado")
        return 1

    if hecha(id_) and "--forzar" not in extra:
        print(f"La etapa {id_} ya esta hecha: {salida}")
        print(f"Para recomputarla:  python correr.py {id_} --forzar")
        return 0

    print("=" * 70)
    print(f" Etapa {id_} — {desc}")
    if gpu:
        print(" Usa GPU. Cerrar otras aplicaciones que consuman VRAM ayuda.")
    print("=" * 70)

    cmd = [sys.executable, str(RAIZ / "src" / script)] + extra
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(RAIZ))
    dt = time.time() - t0

    if r.returncode == 0:
        print(f"\nEtapa {id_} terminada en {formato_duracion(dt)}.")
        sig = [e[0] for e in ETAPAS
               if not hecha(e[0]) and all(hecha(d) for d in e[3]) and e[0] != "09"]
        if sig:
            print(f"Siguiente:  python correr.py {sig[0]}")
    else:
        print(f"\nLa etapa {id_} fallo (codigo {r.returncode}) tras {formato_duracion(dt)}.")
        print(f"Revisar el log: {cfg.DIR_LOGS}")
    return r.returncode


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("estado", "-e", "--estado"):
        estado()
        return 0
    if args[0] in ("ayuda", "-h", "--help"):
        print(__doc__)
        return 0

    if args[0] == "siguiente":
        sig = estado()
        return ejecutar(sig, args[1:]) if sig else 0

    if args[0] == "todo":
        # Encadena las pendientes. Se detiene en cuanto una falla, para no
        # construir resultados encima de una etapa rota.
        while True:
            sig = next((e[0] for e in ETAPAS
                        if not hecha(e[0]) and all(hecha(d) for d in e[3])
                        and e[0] != "09"), None)
            if sig is None:
                print("\nNo quedan etapas pendientes.")
                return 0
            if ejecutar(sig, args[1:]) != 0:
                return 1

    return ejecutar(args[0], args[1:])


if __name__ == "__main__":
    sys.exit(main())
