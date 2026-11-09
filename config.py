# -*- coding: utf-8 -*-
"""
config.py — Unica fuente de verdad del pipeline. VERSION LOCAL.

Adaptada para ejecutar en el equipo "blue": Windows, 16 GB de RAM y una
RTX 4050 portatil con 6 GB de VRAM. Las diferencias con la version de
APOLO estan marcadas con [LOCAL] y son de tres tipos:

  1. RUTAS. Los archivos intermedios y los resultados NO se escriben en la
     carpeta de OneDrive: van a DIR_TRABAJO, fuera de la carpeta
     sincronizada. Escribir varios GB de parquet dentro de OneDrive
     dispararia una subida continua y ademas acerca peligrosamente el
     limite de 260 caracteres de ruta de Windows.
  2. MEMORIA. Con 16 GB no se puede cargar 1.44M de tuits de golpe. Las
     etapas 01 y 02 procesan y escriben por bloques; el resto lee solo las
     columnas que necesita.
  3. GPU. Con 6 GB de VRAM los lotes bajan y la precision mixta pasa a ser
     obligatoria, no opcional.


Paper: "Trust in COVID-19 Information Sources in Colombia: A Cross-Source
Comparison of iCARE Survey Responses and Twitter Discourse, 2020"

Todo parametro, ruta, categoria o umbral vive AQUI. Ningun script del
pipeline define constantes propias. Esto responde directamente al
requisito del asesor: "todas las figuras y tablas deben salir de un
unico script reproducible".

Para mover el proyecto a APOLO basta con copiar la carpeta completa:
las rutas se resuelven relativas a este archivo.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. RUTAS
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent

# [LOCAL] Los datos de entrada no se duplican: son 563 MB de .tab que ya
# estan en el arbol del proyecto. Se busca en varios sitios conocidos y se
# usa el primero que exista, asi el proyecto se puede mover de carpeta sin
# tener que reconfigurar nada.
#
# Para forzar una ubicacion concreta:  setx ICARE_DIR_DATOS "D:\\datos"
def _localizar_datos() -> Path:
    forzada = os.environ.get("ICARE_DIR_DATOS")
    if forzada:
        return Path(forzada)
    candidatos = [
        RAIZ / "datos",                                              # aqui mismo
        RAIZ.parent / "datos",                                       # .../Codigo/datos
        RAIZ.parent / "paperConfianzaPolaridad" / "datos",           # hermana
        RAIZ.parent / "Nueva versión" / "paperConfianzaPolaridad" / "datos",
        RAIZ.parent.parent / "datos",
    ]
    for c in candidatos:
        if (c / "dfTwitter").is_dir():
            return c
    return candidatos[0]   # no existe; se avisara al comprobar el entorno

DIR_DATOS = _localizar_datos()
DIR_TWITTER = DIR_DATOS / "dfTwitter"
ARCHIVO_ICARE = DIR_DATOS / "df_icare_original.xlsx"
ARCHIVO_DICCIONARIO = DIR_DATOS / "diccionario de icare.xlsx"

DIR_RECURSOS = RAIZ / "recursos"
ARCHIVO_CUENTAS_CONOCIDAS = DIR_RECURSOS / "cuentas_conocidas.csv"

# [LOCAL] Directorio de trabajo: la propia carpeta del proyecto, para que
# codigo, datos y resultados vivan juntos.
#
# DOS COSAS QUE HAY QUE SABER AL TENERLO DENTRO DE ONEDRIVE
#
# 1. LONGITUD DE RUTA. Windows corta en 260 caracteres. Con el nombre
#    original de la carpeta ("paperConfianzaPolaridad - local") la ruta
#    mas larga que genera el pipeline media 254: seis de margen. Por eso
#    la carpeta se llama ahora "pipeline-local", que deja unos 25.
#    crear_directorios() avisa si alguna ruta se acerca al limite.
#
# 2. SINCRONIZACION. Los .parquet intermedios suman varios GB y OneDrive
#    los subira. Si molesta, hay dos salidas:
#      - pausar la sincronizacion mientras corre el pipeline, o
#      - sacar SOLO los intermedios, que son regenerables, con la
#        variable de entorno ICARE_DIR_INTERMEDIOS:
#            setx ICARE_DIR_INTERMEDIOS "C:\\icare_intermedios"
#        Los resultados (tablas, figuras, metricas) se quedan aqui.
DIR_TRABAJO = Path(os.environ.get("ICARE_DIR_TRABAJO", RAIZ))

# Salidas. Se crean solas en tiempo de ejecucion.
DIR_INTERMEDIOS = Path(os.environ.get("ICARE_DIR_INTERMEDIOS",
                                      DIR_TRABAJO / "intermedios"))
DIR_RESULTADOS = DIR_TRABAJO / "resultados"
DIR_TABLAS = DIR_RESULTADOS / "tablas"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
DIR_METRICAS = DIR_RESULTADOS / "metricas"
DIR_FUNNEL = DIR_RESULTADOS / "funnel"
DIR_LOGS = DIR_RESULTADOS / "logs"
DIR_VALIDACION = DIR_RESULTADOS / "validacion"

DIRECTORIOS_SALIDA = [
    DIR_INTERMEDIOS, DIR_RESULTADOS, DIR_TABLAS, DIR_FIGURAS,
    DIR_METRICAS, DIR_FUNNEL, DIR_LOGS, DIR_VALIDACION,
]

# Checkpoints de cada etapa
PQ_TWITTER_CRUDO = DIR_INTERMEDIOS / "s01_twitter_crudo.parquet"
PQ_TWITTER_PREP = DIR_INTERMEDIOS / "s02_twitter_preprocesado.parquet"
PQ_SEED1 = DIR_INTERMEDIOS / "s03a_seed_gazetteer.parquet"
PQ_SEED2 = DIR_INTERMEDIOS / "s03b_seed_modelos_clasicos.parquet"
PQ_SEED2B = DIR_INTERMEDIOS / "s03b_seed_transformer.parquet"
PQ_SPLIT_GEO = DIR_INTERMEDIOS / "s03b_particion_geo.parquet"
PQ_SEED3 = DIR_INTERMEDIOS / "s03c_seed_grafo.parquet"
PQ_COLOMBIA = DIR_INTERMEDIOS / "s03d_corpus_colombia.parquet"
PQ_ENTIDADES = DIR_INTERMEDIOS / "s04_entidades.parquet"
PQ_MENCIONES = DIR_INTERMEDIOS / "s04_menciones.parquet"
PQ_POLARIDAD = DIR_INTERMEDIOS / "s05_polaridad.parquet"
PQ_ANALITICO = DIR_INTERMEDIOS / "s05_corpus_analitico.parquet"
PQ_ICARE = DIR_INTERMEDIOS / "s06_icare.parquet"

# ---------------------------------------------------------------------------
# 2. VENTANAS TEMPORALES
# ---------------------------------------------------------------------------
# HALLAZGO CRITICO (verificado 2026-08-31 sobre df_icare_original.xlsx):
# las variables trustinf_* SOLO existen en las olas 5 y 6, recolectadas
# entre 2020-09-16 y 2020-12-06. El corpus de Twitter cubre 2020-03-31 a
# 2020-08-18. NO HAY SOLAPAMIENTO TEMPORAL: 0 respuestas con trustinf
# caen dentro de la ventana de Twitter.
#
# Decision editorial tomada: el paper declara DOS periodos distintos y
# elimina toda afirmacion de "ventana comun". Por eso NO se filtra iCARE
# por la ventana de Twitter; se usa el total de quien respondio trustinf.
TWITTER_FECHA_INICIO = "2020-03-31"
TWITTER_FECHA_FIN = "2020-08-18"

# Solo informativo: se reporta en la Tabla 1 para trazabilidad del "2,130"
# que aparecia en la version previa del manuscrito.
ICARE_VENTANA_REFERENCIA = ("2020-03-31", "2020-08-18")
ICARE_FILTRAR_POR_VENTANA = False   # ver nota arriba

IDIOMA_OBJETIVO = "es"

# ---------------------------------------------------------------------------
# 3. CATEGORIAS DE FUENTE DE INFORMACION
# ---------------------------------------------------------------------------
# Cada categoria es la unidad de comparacion entre iCARE y Twitter/X.
# 'trustinf' es el item de la encuesta; None = la categoria NO tiene item
# de confianza y por tanto queda FUERA de la comparacion cross-source.
# 'observable_twitter' = si la categoria puede identificarse en Twitter.
#
# Esta tabla es la respuesta formal al requisito 2.3.6 del asesor
# ("definir reglas de mapeo de cuentas y menciones a las categorias iCARE").
CATEGORIAS = {
    "lider_nacional": {
        "etiqueta_en": "National political leader",
        "trustinf": "trustinf_sq001",
        "infoso": 6,
        "observable_twitter": True,
    },
    "lider_regional": {
        "etiqueta_en": "Regional political leader",
        "trustinf": "trustinf_sq002",
        "infoso": None,
        "observable_twitter": True,
    },
    "autoridad_salud": {
        "etiqueta_en": "Local health authorities and government",
        "trustinf": "trustinf_sq003",
        "infoso": 5,
        "observable_twitter": True,
    },
    "profesional_salud": {
        "etiqueta_en": "Doctor or healthcare professional",
        "trustinf": "trustinf_sq004",
        "infoso": 4,
        "observable_twitter": True,
    },
    "oms": {
        "etiqueta_en": "World Health Organization",
        "trustinf": "trustinf_sq005",
        "infoso": 8,
        "observable_twitter": True,
    },
    "ciencia": {
        "etiqueta_en": "Scientists and scientific organizations",
        "trustinf": "trustinf_sq006",
        "infoso": 9,
        "observable_twitter": True,
    },
    "famosos": {
        "etiqueta_en": "Famous people",
        "trustinf": "trustinf_sq007",
        "infoso": None,
        "observable_twitter": True,
    },
    "lider_comunitario": {
        "etiqueta_en": "Community, religious or cultural leader",
        "trustinf": "trustinf_sq008",
        "infoso": 7,
        "observable_twitter": True,
    },
    "allegados": {
        # Item de encuesta sin contraparte digital identificable: las
        # cuentas de amigos/familia no son distinguibles en Twitter.
        "etiqueta_en": "Friends, family, colleagues",
        "trustinf": "trustinf_sq009",
        "infoso": 1,
        "observable_twitter": False,
    },
    "medios": {
        # Contraparte digital sin item de confianza en iCARE. Se reporta
        # en los resultados de Twitter pero NO entra en la Figura de
        # concordancia cross-source.
        "etiqueta_en": "News media",
        "trustinf": None,
        "infoso": 3,
        "observable_twitter": True,
    },
}

# Categorias que entran en la comparacion cross-source (Figura 5):
# necesitan item de confianza Y ser observables en Twitter.
CATEGORIAS_CROSS_SOURCE = [
    k for k, v in CATEGORIAS.items()
    if v["trustinf"] is not None and v["observable_twitter"]
]

CATEGORIAS_TWITTER = [k for k, v in CATEGORIAS.items() if v["observable_twitter"]]

ETIQUETA_EN = {k: v["etiqueta_en"] for k, v in CATEGORIAS.items()}

# Etiquetas "sin clasificar" que no son categorias sustantivas
SIN_CATEGORIA = "sin_categoria"
SIN_DESCRIPCION = "sin_descripcion"

# ---------------------------------------------------------------------------
# 4. iCARE — codificacion
# ---------------------------------------------------------------------------
# Escala trustinf (segun 'diccionario de icare.xlsx'):
#   1 = I trust to a great extent
#   2 = I trust somewhat
#   3 = I do not trust
#   4 = I do not trust at all
#   5 = I don't know / I prefer not to answer
#
# DECISION: 1,2 -> 1 (confia) ; 3,4 -> 0 (no confia) ; 5 -> MISSING.
# El 5 es no-respuesta, no desconfianza: contarlo como 0 inflaria
# artificialmente el distrust. El script reporta cuantos 5 hubo por item
# para que la decision sea auditable.
TRUST_VALORES_SI = (1, 2)
TRUST_VALORES_NO = (3, 4)
TRUST_VALORES_MISSING = (5,)

ICARE_COL_FECHA = "startdate"      # submitdate tiene 502 nulos; startdate 0
ICARE_COL_OLA = "wave"

# Variables sociodemograficas de la Tabla 1 y su etiquetado
# Etiquetas VERIFICADAS contra 'diccionario de icare.xlsx' (2026-08-31).
# No inferir estas etiquetas: el diccionario es la fuente. Un error aqui
# se propaga directo a la Tabla 1 del manuscrito.
ICARE_SOCIODEMOGRAFICAS = {
    "sex": {
        # dicc.: "How would you describe your sex?"
        "etiqueta_en": "Sex",
        "niveles": {1: "Man", 2: "Woman", 3: "Other", 4: "Prefer not to answer"},
    },
    "edu": {
        # dicc.: "What is your highest level of completed education?"
        "etiqueta_en": "Education (highest completed)",
        "niveles": {
            1: "Primary or elementary school or less",
            2: "Secondary or high school",
            3: "College or University degree",
            4: "Graduate or Postgraduate degree",
            5: "Never been to school",
            6: "Do not know / prefer not to answer",
        },
    },
}
ICARE_COL_EDAD = "age"
ICARE_GRUPOS_EDAD = [(18, 29), (30, 39), (40, 49), (50, 59), (60, 200)]

# emplstat_sq001..sq008 es una pregunta de SELECCION MULTIPLE (0/1 por
# opcion), no una variable categorica unica. Por eso en el manuscrito
# anterior el denominador de Employment sumaba 164: se estaban sumando
# personas que marcaron una sola casilla. Se reporta como multi-respuesta,
# con denominador = quienes contestaron el bloque.
ICARE_EMPLEO_PREFIJO = "emplstat_sq"
# Etiquetas VERIFICADAS contra el diccionario. Ojo: `emplstat` es la
# situacion laboral PREVIA a la pandemia ("Prior to the COVID-19
# pandemic, how would you describe your employment status?"), no la
# actual. La Tabla 1 debe decirlo. La variable de situacion actual
# (`cempstat_*`) no esta presente en este extracto de Colombia.
#
# El diccionario trae una fila extra sin nombre de variable entre sq003 y
# sq004 ([Unemployed]); el bloque paralelo `cempstat_*`, que si esta
# completo y bien alineado, confirma que sq003 es la casilla de desempleo.
ICARE_EMPLEO_ETIQUETA = "Employment status prior to the pandemic (multiple response)"
ICARE_EMPLEO_NIVELES = {
    "emplstat_sq001": "Retired or homemaker",
    "emplstat_sq002": "Receiving support from a government program",
    "emplstat_sq003": "Unemployed / on unemployment insurance",
    "emplstat_sq004": "Student",
    "emplstat_sq005": "Part-time job",
    "emplstat_sq006": "Full-time job",
    "emplstat_sq007": "Self-employed",
    "emplstat_sq008": "Do not know / prefer not to answer",
}

# ---------------------------------------------------------------------------
# 5. MODELOS  [LOCAL: lotes calibrados para 6 GB de VRAM]
# ---------------------------------------------------------------------------
# La RTX 4050 portatil tiene 6 GB. Todo lo que pasa por GPU usa precision
# mixta (fp16), que aqui no es una optimizacion opcional sino la condicion
# para que los modelos quepan. Los lotes estan puestos con margen: si aun
# asi aparece "CUDA out of memory", basta con reducirlos a la mitad.
USAR_FP16 = True

# Polaridad: modelo exacto, version fijada. Responde al requisito 2.3.7.
MODELO_POLARIDAD = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MODELO_POLARIDAD_REVISION = "main"
POLARIDAD_ETIQUETAS = {0: "Negative", 1: "Neutral", 2: "Positive"}
POLARIDAD_BATCH = 64          # APOLO: 128
POLARIDAD_MAX_LEN = 128

# NOTA METODOLOGICA (decision tomada, debe quedar escrita en Methods):
# la polaridad se calcula sobre el TUIT COMPLETO, no sobre el par
# (tuit, entidad). No es target-dependent sentiment. Por tanto el paper
# NO puede afirmar "sentiment toward an entity"; solo
# "polarity of the discourse in which the entity is mentioned".
POLARIDAD_ES_TARGET_DEPENDENT = False

# Entidades: clasificacion zero-shot sobre la descripcion del autor.
#
# CUIDADO CON EL LOTE. El pipeline de zero-shot evalua cada texto contra
# CADA hipotesis, asi que el numero real de secuencias por paso es
# ENTIDADES_BATCH x 9. Con 4 son 36 secuencias de un modelo "large": es lo
# que cabe comodo en 6 GB.
MODELO_ENTIDADES = "joeddav/xlm-roberta-large-xnli"
ENTIDADES_BATCH = 4           # APOLO: 32
# Cuantas cuentas entran en la tabla de las mas mencionadas del corpus
# colombiano (s04_top_menciones_corpus.csv y la figura correspondiente).
TOP_N_MENCIONES = 10

ENTIDADES_UMBRAL_MIN = 0.30
ENTIDADES_MARGEN_MIN = 0.05

# Alternativa mas ligera y rapida si el zero-shot resulta demasiado lento:
# mDeBERTa-v3-base (280M frente a 560M) con resultados comparables o
# mejores en NLI multilingue. Para usarla, poner su nombre en
# MODELO_ENTIDADES y subir ENTIDADES_BATCH a 16.
#   MODELO_ENTIDADES = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
#
# El modelo que se acabe usando es el que hay que declarar en Methods.
MODELO_ENTIDADES_ALTERNATIVO = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Modelo transformer para el clasificador geografico (5o modelo)
MODELO_GEO_BERT = "dccuchile/bert-base-spanish-wwm-cased"
GEO_BERT_EPOCAS = 2
GEO_BERT_BATCH = 16           # APOLO: 32
GEO_BERT_BATCH_INFERENCIA = 64
GEO_BERT_MAX_LEN = 96

# ---------------------------------------------------------------------------
# 6. CLASIFICACION GEOGRAFICA
# ---------------------------------------------------------------------------
SEMILLA = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# Umbral por defecto. El script tambien calcula el umbral que maximiza F1
# en validacion y reporta AMBOS (el fijo era 0.85 en la version antigua,
# sin justificacion documentada).
GEO_UMBRAL_FIJO = 0.85
GEO_OPTIMIZAR_UMBRAL = True

# Regla de combinacion final (requisito 2.3.5 del asesor).
# Un tuit se marca como de origen colombiano si cumple la regla elegida.
#   "mayoria"  -> >= GEO_MIN_VOTOS de las 3 evidencias (gazetteer, ML, grafo)
#   "union"    -> cualquiera de las 3
#   "gazetteer_o_ml" -> gazetteer o modelos ML (grafo solo expande)
GEO_REGLA_COMBINACION = "mayoria"
GEO_MIN_VOTOS = 2

# El voto del bloque ML se decide por mayoria entre los 5 modelos
GEO_ML_MIN_MODELOS = 3

# Evidencia 3: propagacion sobre el grafo de respuestas
GRAFO_ITERACIONES = 2          # rondas de propagacion
GRAFO_MIN_INTERACCIONES = 2    # respuestas minimas para considerar un vinculo
GRAFO_UMBRAL_VECINOS = 0.60    # proporcion de vecinos colombianos para marcar

TFIDF_NGRAMAS = (1, 3)
TFIDF_MAX_FEATURES = 20000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.85

# ---------------------------------------------------------------------------
# 6bis. GAZETTEER: EXCLUSIONES POR AMBIGUEDAD
# ---------------------------------------------------------------------------
# El gazetteer heredado marca como colombiano cualquier texto que contenga
# el nombre de un municipio. En un corpus de COVID-19 en espanol de toda
# Hispanoamerica eso produce falsos positivos masivos, porque muchos
# municipios colombianos son homonimos de:
#   (a) palabras corrientes del espanol  -> "fundacion", "une", "toca",
#       "cesar", "meta", "pasto", "soledad", "bello", "piedras"
#   (b) apellidos frecuentes             -> "gonzalez", "ortega", "miranda",
#       "restrepo", "rivera", "morales"
#   (c) lugares mas conocidos de otros paises -> "madrid", "santiago",
#       "california", "florida", "cordoba", "la paz", "armenia", "valencia"
#
# Medido sobre una muestra de 19,404 tuits, de los veinte terminos mas
# activados solo cinco eran inequivocamente colombianos.
#
# ESTA LISTA NO ES DEFINITIVA. La etapa 03a emite
# resultados/tablas/s03a_terminos_mas_frecuentes.csv sobre el corpus
# completo: revisarlo y ampliar estas listas es parte del trabajo, y el
# hecho de que sea auditable es justamente lo que pide el asesor.
TOPONIMOS_AMBIGUOS_EXCLUIDOS = {
    # palabras comunes del espanol
    "fundacion", "une", "toca", "cesar", "meta", "pasto", "soledad", "bello",
    "piedras", "la pena", "concepcion", "la esperanza", "la mesa", "la union",
    "la victoria", "la paz", "la cruz", "la sierra", "la playa", "la vega",
    "la palma", "la merced", "la cumbre", "la gloria", "la primavera",
    "la salina", "la belleza", "la ceja", "la florida", "la tola", "la argentina",
    "el paso", "el dorado", "el retiro", "el rosal", "el carmen", "el tambo",
    "el cerrito", "el colegio", "el molino", "el roble", "el socorro",
    "el santuario", "el espinal", "el espino", "el guamo", "el retorno",
    "providencia", "victoria", "candelaria", "concordia", "primavera",
    "esperanza", "manta", "toro", "peque", "andes", "guadalupe", "salgar",
    "atlantico", "magdalena", "choco", "cota", "iza", "gama", "tota", "chita",
    "sopo", "tena", "guaca", "supia", "guapi", "buenavista", "buenos aires",
    "campoalegre", "cabrera", "balboa", "puerto rico", "la apartada",
    "san isidro", "el copey",
    # apellidos frecuentes que tambien son municipios
    "gonzalez", "ortega", "miranda", "restrepo", "rivera", "morales", "bolivar",
    "santander", "narino", "nariño", "sucre", "cordoba", "caldas", "silvia",
    "tello", "uribia", "la uribe", "elias", "gutierrez", "guerrero",
    # lugares cuyo referente dominante esta fuera de Colombia
    "madrid", "santiago", "california", "florida", "valparaiso", "andalucia",
    "malaga", "monterrey", "sevilla", "granada", "toledo", "valencia",
    "san francisco", "san antonio", "san jose", "san juan", "san pablo",
    "san pedro", "san carlos", "san luis", "san marcos", "san martin",
    "san rafael", "san miguel", "san fernando", "san benito", "san lorenzo",
    "santa rosa", "santa maria", "santa barbara", "santa ana", "santa catalina",
    "santa isabel", "santacruz", "santa cruz", "colon", "belen", "jerusalen",
    "palestina", "libano", "medina", "trinidad", "argelia", "versalles",
    "florencia", "armenia", "cartago", "palmira", "amazonas", "roma", "italia",
    "corinto", "betania", "chile", "brasil",
}

# Marcadores de jerga que en este corpus NO discriminan origen colombiano.
# "chino/china/chinos/chinas" significa "nino" en Colombia, pero en un
# corpus de COVID-19 se refiere abrumadoramente a China: sumaron 396
# activaciones en la muestra de prueba, practicamente todas falsos
# positivos. El resto son verbos o sustantivos del espanol general.
JERGA_AMBIGUA_EXCLUIDA = {
    "china", "chino", "chinos", "chinas", "manejar", "metido", "metida",
    "metidos", "metidas", "filo", "filos", "lacra", "trago", "balde", "caneca",
    "mota", "chepa", "chiripa", "pola", "polas", "boleta", "melo", "peye",
    "pulido", "pulida", "pulidos", "pulidas", "nero", "nera", "neros", "neras",
    "cucho", "cucha", "cuchos", "cuchas", "guache", "guaches", "santiago",
}

# Longitud minima de un termino de UNA sola palabra en el nivel extendido.
# Descarta los homonimos cortisimos que sobreviven a la lista de exclusion.
# No se aplica a terminos con espacios: "san jose del guaviare" es
# inequivoco por su propia extension.
GAZETTEER_LONGITUD_MINIMA_EXTENDIDO = 6

# El nivel extendido (municipios, pueblos, parques) NO basta por si solo:
# exige otra senal (nucleo, jerga o Location = 'CO'). Poner False
# reproduce el comportamiento plano de la version 1.
GAZETTEER_EXTENDIDO_EXIGE_CORROBORACION = True

# ---------------------------------------------------------------------------
# 7. ESTADISTICA
# ---------------------------------------------------------------------------
ALFA = 0.05
IC_METODO = "wilson"        # intervalo de confianza para proporciones
POSTHOC_CORRECCION = "bonferroni"
BOOTSTRAP_N = 5000

# ---------------------------------------------------------------------------
# 8. VALIDACION MANUAL
# ---------------------------------------------------------------------------
# Muestras que el script genera para anotacion humana. Pendiente de anotar:
# mientras no se devuelvan anotadas, el paper debe declararlo como limitacion.
VALIDACION_N_POLARIDAD = 400
VALIDACION_N_ENTIDAD = 400
VALIDACION_N_GEO = 400
VALIDACION_DOBLE_ANOTADOR = False   # True -> genera copia para 2o anotador

# ---------------------------------------------------------------------------
# 9. EJECUCION  [LOCAL]
# ---------------------------------------------------------------------------
import multiprocessing as _mp

# Tamano de bloque para las etapas con checkpoint. Con 16 GB de RAM,
# 50,000 tuits por bloque mantienen el consumo por debajo de ~2 GB en la
# etapa mas pesada. En APOLO era 200,000.
CHUNK_TWEETS = 50_000

# Filas por grupo al escribir parquet. Grupos pequenos permiten leer el
# archivo por partes sin cargarlo entero.
PARQUET_ROW_GROUP = 50_000

# Archivos .tab que se leen antes de volcar a disco en la etapa 01.
LOTE_ARCHIVOS_TAB = 20

# Procesos para spaCy y sklearn. En Windows cada proceso de spaCy recarga
# el modelo, asi que conviene no pasarse: se dejan dos nucleos libres.
N_PROCESOS = max(1, min(6, (_mp.cpu_count() or 4) - 2))

MODO_PRUEBA = False
N_PRUEBA_ARCHIVOS = 4

# [LOCAL] Limite de memoria orientativo, solo para los avisos del log.
RAM_DISPONIBLE_GB = 16

LIMITE_RUTA_WINDOWS = 260
MARGEN_AVISO = 25


def crear_directorios():
    """Crea toda la estructura de salida. Idempotente.

    Ademas, en Windows comprueba que las rutas que va a generar el
    pipeline no se acerquen al limite de 260 caracteres. Es un fallo que
    aparece a mitad de una corrida larga y con un mensaje que no dice
    nada util, asi que vale la pena detectarlo al principio.
    """
    for d in DIRECTORIOS_SALIDA:
        d.mkdir(parents=True, exist_ok=True)

    import sys
    if not sys.platform.startswith("win"):
        return
    # el nombre mas largo que llega a generar el pipeline
    peor = max(
        len(str(DIR_TABLAS / "s07_tabla3_entidad_polaridad_menciones.csv")),
        len(str(DIR_FIGURAS / "s03b_matriz_confusion_random_forest.png")),
        len(str(DIR_VALIDACION / "muestra_polaridad_anotador2.xlsx")),
        len(str(DIR_INTERMEDIOS / "s02_bloques" / "bloque_9999.parquet")),
    )
    if peor > LIMITE_RUTA_WINDOWS - MARGEN_AVISO:
        print()
        print("=" * 70)
        print(f" AVISO: la ruta mas larga del pipeline mide {peor} caracteres")
        print(f" y Windows corta en {LIMITE_RUTA_WINDOWS}. Quedan {LIMITE_RUTA_WINDOWS - peor}.")
        print()
        print(" Si alguna etapa falla con un error de archivo no encontrado o")
        print(" ruta invalida, es esto. Soluciones, de mas simple a menos:")
        print("   - mover el proyecto a una carpeta con ruta mas corta")
        print("   - sacar los intermedios:")
        print('       setx ICARE_DIR_INTERMEDIOS "C:\\icare_intermedios"')
        print("=" * 70)
        print()
