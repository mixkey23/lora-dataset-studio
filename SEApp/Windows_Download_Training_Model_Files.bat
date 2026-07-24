@echo off
setlocal

set "ROOT_DIR=%~dp0"

cd /d "%ROOT_DIR%SECourses_Musubi_Trainer"
if errorlevel 1 goto :environment_error

call .\venv\Scripts\activate.bat
if errorlevel 1 goto :environment_error

cd /d "%ROOT_DIR%"
if errorlevel 1 goto :environment_error

echo.
echo ====================================================
echo Training Models Download
echo ====================================================
echo.
python Download_Train_Models.py

set "DOWNLOAD_EXIT=%ERRORLEVEL%"

if not "%DOWNLOAD_EXIT%"=="0" (
    echo.
    echo One or more model files failed. Re-run this script to resume the download.
)

pause
exit /b %DOWNLOAD_EXIT%

:dependency_error
echo.
echo Failed to install or update the download dependencies.
echo Check the network connection and the error shown above, then run this script again.
pause
exit /b 1

:environment_error
echo.
echo Could not find or activate SECourses_Musubi_Trainer\venv.
echo Run Windows_Install_and_Update.bat first, then try again.
pause
exit /b 1
