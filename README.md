# iCARE × Twitter/X: COVID-19 information sources in Colombia

Analysis code for the article

> Puerta Velez AF, Lemos Hoyos M, Suarez Sierra BM, Jaramillo Giraldo V,
> Hincapié Montoya EM, on behalf of the iCARE team. *Comparing Twitter/X
> discourse with survey trust in COVID-19 information sources in Colombia: the
> role of classifier precision.* Submitted to the Journal of Computational
> Social Science.

The pipeline builds a corpus of Colombian tweets about COVID-19 (31 March –
18 August 2020), classifies the information-source category of the accounts
cited in each tweet and the polarity of each tweet, validates the three
automatic classifiers against blind human annotation, and compares the
resulting ordering of negative discourse with self-reported distrust in the
iCARE survey. Every table and figure in the article is produced by the
scripts in this repository; all parameters, thresholds and paths live in a
single file, `config.py`.

Code comments and console messages are in Spanish.

## Data

The data are **not** included. The Twitter/X corpus is a public dataset;
the iCARE survey microdata are restricted. See [`datos/README.md`](datos/README.md)
for where to obtain each one and where to place it.

## Repository structure

```
config.py              single source of parameters, paths and thresholds
correr.py              stage launcher (dependency-aware, resumable)
correr.bat             Windows wrapper that activates the conda environment
entorno.yml            conda environment
requirements.txt       pip requirements (same pins)
src/                   one script per pipeline stage
recursos/
  gazetteer.py         two-tier Colombian gazetteer, exclusions and lexicon
  cuentas_conocidas.csv  curated dictionary of 56 public accounts mapped to
                         iCARE categories, with a justification for each
docs/
  CODEBOOK_anotacion.md  annotation codebook used for the human validation
datos/                 input data go here (not distributed)
```

Outputs are written at run time to `intermedios/` (Parquet checkpoints) and
`resultados/` (`tablas/` CSV, `figuras/` PNG and PDF at 300 dpi, `metricas/`
JSON, `logs/`).

## Pipeline stages

| Stage | Script | What it does |
|---|---|---|
| 01 | `s01_cargar_twitter.py` | Loads the 290 `.tab` files, filters language and window, removes exact duplicates, reports coverage |
| 02 | `s02_preprocesar.py` | Extracts mentions and hashtags; builds raw, normalized and lemmatized text |
| 03a | `s03a_gazetteer.py` | Geographic evidence 1: gazetteer and Colombian-Spanish lexicon |
| 03b | `s03b_modelos_clasicos.py` | Geographic evidence 2: Naive Bayes, linear SVM, random forest, XGBoost (distant supervision) |
| 03bt | `s03b_modelo_transformer.py` | Geographic evidence 2: BETO fine-tuning (GPU) |
| 03c | `s03c_grafo.py` | Geographic evidence 3: reply-graph label propagation |
| 03d | `s03d_corpus_colombia.py` | Majority rule (2 of 3) → Colombian corpus |
| 04 | `s04_entidades.py` | Entity cascade: curated dictionary, lexical rules, zero-shot classification (GPU) |
| 05 | `s05_polaridad.py` | Tweet polarity with XLM-RoBERTa (GPU) |
| 06 | `s06_icare.py` | iCARE sample, trust items and sociodemographic tables (needs iCARE data) |
| 07 | `s07_estadistica.py` | Entity–polarity association, confidence intervals, residuals, cross-source comparison |
| 07b | `s07b_consolidar_validar.py` | Consolidates values and internal consistency checks |
| 09 | `s09_evaluar_anotacion.py` | Precision, Cohen's κ and error-adjusted polarity from the human annotation |
| 08 | `s08_figuras.py` | All figures, from the CSV tables only |
| 10 | `s10_sensibilidad.py` | Sensitivity analyses (high-precision geographic corpus, curated-dictionary corpus) |

Pretrained models (downloaded automatically from Hugging Face):
`dccuchile/bert-base-spanish-wwm-cased` (stage 03bt),
`joeddav/xlm-roberta-large-xnli` (stage 04) and
`cardiffnlp/twitter-xlm-roberta-base-sentiment` (stage 05).

## Installation

Developed and run on Windows with Python 3.11, 16 GB RAM and an NVIDIA GPU with 6 GB
of VRAM.

```bash
conda env create -f entorno.yml
conda activate icare
python -m spacy download es_core_news_sm

# PyTorch with CUDA is installed separately (the conda build is usually CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install accelerate

python -c "import torch; print(torch.cuda.is_available())"
```

Without a GPU the pipeline still runs, but stages 03bt, 04 and 05 take days
instead of hours (`--cpu` forces CPU).

## Running

```bash
python correr.py estado      # status of every stage
python correr.py 01          # run one stage
python correr.py siguiente   # run the first pending stage
python correr.py todo        # run all pending stages in order
```

A stage is not re-run if its checkpoint exists, unless `--forzar` is given.
`--prueba` processes only 4 `.tab` files, which checks the installation in
minutes. A quick check that needs no Twitter data:

```bash
python correr.py 06          # should report 2301, 2129, 125 and 0 in s06_flujo_icare.csv
```

## Human validation

Stage 09 reads the annotated validation samples (polarity, entity category and
geographic origin; about 400 cases each, two independent annotators plus
adjudication). These files contain tweet text and are not distributed; they
are available from the corresponding author on reasonable request; if obtained,
place them in `resultados/validacion/anotados/`. The
annotation rules are in [`docs/CODEBOOK_anotacion.md`](docs/CODEBOOK_anotacion.md).
Without them, stage 09 and the outputs that depend on validated precision
cannot be reproduced; every other Twitter/X stage can.

## Citation

Please cite the article and this software (see [`CITATION.cff`](CITATION.cff)).

## License

MIT (see [`LICENSE`](LICENSE)). The iCARE data and the Twitter/X dataset are
subject to their own terms.
