@echo off

set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy

echo WARNING. For this auto installer to work you need to have installed Python 3.12.10 (may work with 3.10.x, 3.11.x, 3.13.x too), Git, FFmpeg, cuDNN 9.17+, CUDA 13.0, Visual Studio Community Edition with All c++ options
echo This tutorial shows all step by step : https://youtu.be/DrhUHnYfwC0?si=UAAVyZ8_QUPAjy7a

git clone --depth 1 https://github.com/FurkanGozukara/SECourses_Musubi_Trainer

cd SECourses_Musubi_Trainer

git reset --hard

git pull

git clone --depth 1 https://github.com/FurkanGozukara/convert_to_quant

cd convert_to_quant

git reset --hard

git pull

cd ..

git clone --depth 1 https://github.com/FurkanGozukara/musubi-tuner

cd musubi-tuner

git reset --hard

git pull

git checkout main

git pull

cd ..

py --version >nul 2>&1
if "%ERRORLEVEL%" == "0" (
    echo Python launcher is available. Generating Python 3.12 VENV
    py -3.12 -m venv venv
) else (
    echo Python launcher is not available, generating VENV with default Python. Make sure that it is 3.12
    python -m venv venv
)

call .\venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_musubi_trainer.txt --index-strategy unsafe-best-match

cd SECourses_Musubi_Trainer

cd musubi-tuner

uv pip install -e .

cd ..

cd convert_to_quant

uv pip install -e .

echo installation completed check for errors

REM Pause to keep the command prompt open
pause