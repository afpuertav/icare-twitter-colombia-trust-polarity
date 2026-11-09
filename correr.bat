@echo off
setlocal EnableDelayedExpansion
REM ===========================================================================
REM  correr.bat - lanzador del pipeline en Windows.
REM
REM  Se encarga de encontrar conda y activar el entorno, para no depender de
REM  que "conda" este en el PATH de la consola. Basta con abrir la consola
REM  normal (cmd) en esta carpeta y escribir:
REM
REM      correr.bat estado
REM      correr.bat 01
REM      correr.bat todo
REM      correr.bat crear-entorno     <- la primera vez, crea el entorno conda
REM
REM  Tambien funciona haciendo doble clic: muestra el estado y espera.
REM ===========================================================================

cd /d "%~dp0"

REM --- localizar conda -------------------------------------------------------
set "CONDA_BAT="
for %%P in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\AppData\Local\miniconda3"
    "%USERPROFILE%\AppData\Local\Continuum\miniconda3"
    "%ProgramData%\miniconda3"
    "%ProgramData%\Anaconda3"
    "C:\miniconda3"
    "C:\Anaconda3"
) do (
    if exist "%%~P\Scripts\activate.bat" (
        if not defined CONDA_BAT set "CONDA_BAT=%%~P\Scripts\activate.bat"
    )
)

if not defined CONDA_BAT (
    echo.
    echo No encuentro conda en las ubicaciones habituales.
    echo Abre "Anaconda Prompt" o "Miniconda Prompt" desde el menu Inicio,
    echo navega a esta carpeta y ejecuta:  python correr.py %*
    echo.
    pause
    exit /b 1
)

REM --- crear el entorno, si se pidio ----------------------------------------
if /i "%~1"=="crear-entorno" (
    echo Creando el entorno conda "icare"...
    call "%CONDA_BAT%"
    call conda env create -f entorno.yml
    if errorlevel 1 (
        echo.
        echo Fallo la creacion del entorno. Si ya existia, borralo primero con:
        echo    conda env remove -n icare
        pause
        exit /b 1
    )
    call conda activate icare

    echo.
    echo Descargando el modelo de spaCy en espanol...
    python -m spacy download es_core_news_sm

    call :instalar_torch

    echo.
    echo === Comprobacion de la GPU ===
    python -c "import torch; ok=torch.cuda.is_available(); print('GPU detectada:', ok); print('Tarjeta:', torch.cuda.get_device_name(0) if ok else 'ninguna'); print('' if ok else 'AVISO: sin GPU las etapas 03bt, 04 y 05 tardarian dias en vez de horas.')"

    echo.
    echo Entorno listo. Siguiente paso:   correr.bat estado
    echo.
    pause
    exit /b 0
)

REM --- reparar la GPU -------------------------------------------------------
if /i "%~1"=="reparar-gpu" (
    call "%CONDA_BAT%"
    call conda activate icare
    if errorlevel 1 (
        echo No existe el entorno "icare". Crearlo con:  correr.bat crear-entorno
        pause
        exit /b 1
    )
    call :instalar_torch
    echo.
    pause
    exit /b 0
)

REM --- reparar spaCy --------------------------------------------------------
if /i "%~1"=="reparar-spacy" (
    call "%CONDA_BAT%"
    call conda activate icare
    if errorlevel 1 (
        echo No existe el entorno "icare". Crearlo con:  correr.bat crear-entorno
        pause
        exit /b 1
    )
    echo.
    echo Version actual:
    python -c "import spacy, srsly, thinc; print('  spacy', spacy.__version__, '| srsly', srsly.__version__, '| thinc', thinc.__version__)"
    echo.
    echo Reinstalando spaCy y su modelo desde pip, para que sean coherentes.
    REM El error "int is not allowed for map key" es un desajuste entre la
    REM version de spaCy y la del modelo. Mezclar el spaCy de conda-forge
    REM con un modelo bajado aparte lo provoca; instalando ambos con pip
    REM la resolucion de versiones los mantiene alineados.
    pip uninstall -y spacy srsly thinc es-core-news-sm >nul 2>&1
    pip install -U spacy
    python -m spacy download es_core_news_sm
    echo.
    echo === Comprobacion ===
    python -m spacy validate
    python -c "import spacy; nlp=spacy.load('es_core_news_sm'); d=nlp('Los ninos jugaban en Medellin'); print('spaCy carga bien. Lemas:', [t.lemma_ for t in d])"
    echo.
    pause
    exit /b 0
)

REM --- reparar el tokenizador -----------------------------------------------
if /i "%~1"=="reparar-tokenizer" (
    call "%CONDA_BAT%"
    call conda activate icare
    if errorlevel 1 (
        echo No existe el entorno "icare". Crearlo con:  correr.bat crear-entorno
        pause
        exit /b 1
    )
    echo Instalando sentencepiece y protobuf, que necesita el tokenizador
    echo de XLM-RoBERTa para las etapas 04 y 05...
    pip install sentencepiece protobuf
    echo.
    echo === Comprobacion ===
    python -c "import sentencepiece, google.protobuf; print('sentencepiece y protobuf disponibles')"
    echo.
    pause
    exit /b 0
)

REM --- activar el entorno ---------------------------------------------------
call "%CONDA_BAT%"
call conda activate icare
if errorlevel 1 (
    echo.
    echo No se pudo activar el entorno conda "icare".
    echo Crearlo con:   correr.bat crear-entorno
    echo.
    pause
    exit /b 1
)

REM --- ejecutar -------------------------------------------------------------
if "%~1"=="" (
    python correr.py estado
    echo.
    pause
) else (
    python correr.py %*
)
endlocal
exit /b %errorlevel%

REM ===========================================================================
REM  Subrutina: instalar PyTorch con CUDA, quitando antes cualquier version
REM  CPU que hubiera entrado como dependencia de otro paquete.
REM
REM  Sin el desinstalado previo, pip ve "torch ya instalado" y no hace nada,
REM  que es como acaba uno con GPU detectada: False.
REM ===========================================================================
:instalar_torch
echo.
echo Quitando cualquier PyTorch previo, que puede ser la version CPU...
pip uninstall -y torch torchvision torchaudio >nul 2>&1

echo Instalando PyTorch con CUDA 12.4. Es una descarga grande, ten paciencia.
pip install torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo.
    echo Fallo con CUDA 12.4. Probando con CUDA 11.8...
    pip install torch --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo Instalando accelerate, ahora que torch ya es el correcto...
pip install accelerate

echo.
echo === Comprobacion de la GPU ===
python -c "import torch; v=torch.__version__; ok=torch.cuda.is_available(); print('PyTorch:', v); print('GPU detectada:', ok); print('Tarjeta:', torch.cuda.get_device_name(0) if ok else 'ninguna'); print(''); print('LISTO: las etapas de GPU usaran la tarjeta.' if ok else 'SIGUE SIN VERSE LA GPU.'); print('' if ok else ('  - Si la version NO lleva +cu, se instalo la de CPU.' if '+cu' not in v else '  - La version es de CUDA, asi que el problema es el driver de NVIDIA.')); print('' if ok else '  - Comprobar el driver con:  nvidia-smi')"
exit /b 0
