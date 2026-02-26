@echo off
setlocal enabledelayedexpansion
:: ─────────────────────────────────────────────────────────────────────────────
:: PINN PDE Solver — Windows Setup & Run
:: ─────────────────────────────────────────────────────────────────────────────
::
:: Usage:
::   setup.bat              Full setup: venv, deps, train, serve
::   setup.bat --serve      Skip training, just start the server
::   setup.bat --train      Only train the model (no server)
::   setup.bat --test       Run the test suite
::   setup.bat --clean      Remove venv and cached files
::
:: ─────────────────────────────────────────────────────────────────────────────

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "MODEL_DIR=%SCRIPT_DIR%models"
set "MODEL_FILE=%MODEL_DIR%\burgers_model.pth"
set "REQUIREMENTS=%SCRIPT_DIR%requirements.txt"
if "%PORT%"=="" set "PORT=8000"

echo.
echo ══════════════════════════════════════════════
echo   PINN PDE Solver — Setup
echo ══════════════════════════════════════════════
echo.

:: ── Parse argument ───────────────────────────────────────────────────────────
set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

if "%MODE%"=="--help" goto :show_help
if "%MODE%"=="-h"     goto :show_help

:: ── Find Python ──────────────────────────────────────────────────────────────
call :find_python
if errorlevel 1 (
    echo [ERR]  Python 3.10+ is required but was not found.
    echo [ERR]  Install it from https://www.python.org/downloads/
    echo [ERR]  Make sure "Add Python to PATH" is checked during installation.
    exit /b 1
)
echo [OK]   Found Python at %PYTHON_CMD%

:: ── Route to mode ────────────────────────────────────────────────────────────
if "%MODE%"=="--serve" goto :mode_serve
if "%MODE%"=="-s"      goto :mode_serve
if "%MODE%"=="--train"  goto :mode_train
if "%MODE%"=="-t"       goto :mode_train
if "%MODE%"=="--test"   goto :mode_test
if "%MODE%"=="--clean"  goto :mode_clean
if "%MODE%"=="-c"       goto :mode_clean

:: Default: full setup
call :ensure_venv
call :install_deps
call :train_model
call :start_server
goto :eof

:mode_serve
call :ensure_venv
call :start_server
goto :eof

:mode_train
call :ensure_venv
call :install_deps
call :train_model
goto :eof

:mode_test
call :ensure_venv
call :install_deps
call :run_tests
goto :eof

:mode_clean
call :clean
goto :eof


:: ═════════════════════════════════════════════════════════════════════════════
:: Functions
:: ═════════════════════════════════════════════════════════════════════════════

:: ── Find Python 3.10+ ────────────────────────────────────────────────────────
:find_python
for %%P in (python py python3) do (
    where %%P >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=*" %%V in ('%%P -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
            for /f "tokens=1,2 delims=." %%A in ("%%V") do (
                if %%A GEQ 3 if %%B GEQ 10 (
                    set "PYTHON_CMD=%%P"
                    exit /b 0
                )
            )
        )
    )
)
exit /b 1

:: ── Create / activate virtual environment ────────────────────────────────────
:ensure_venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    echo [OK]   Virtual environment created
)
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK]   Activated venv
exit /b 0

:: ── Install dependencies ─────────────────────────────────────────────────────
:install_deps
echo [INFO] Installing dependencies...
pip install --upgrade pip --quiet
pip install -r "%REQUIREMENTS%" --quiet
echo [OK]   All dependencies installed
exit /b 0

:: ── Train the model ──────────────────────────────────────────────────────────
:train_model
if exist "%MODEL_DIR%\*.pth" (
    echo [WARN] Trained models already exist.
    set /p "ANSWER=       Retrain all from scratch? [y/N] "
    if /i not "!ANSWER!"=="y" (
        echo [INFO] Skipping training
        exit /b 0
    )
)
echo [INFO] Training PDE models (this may take several minutes)...
python "%SCRIPT_DIR%pinn_model.py" --equation all --no-show-plot --save-plot
echo [OK]   Training complete — models saved to models/
exit /b 0

:: ── Start the server ─────────────────────────────────────────────────────────
:start_server
if not exist "%MODEL_DIR%\*.pth" (
    echo [ERR]  No trained models found. Run training first: setup.bat --train
    exit /b 1
)
echo [INFO] Starting server on http://127.0.0.1:%PORT%
echo.
echo   ┌──────────────────────────────────────────────┐
echo   │  Dashboard:  http://127.0.0.1:%PORT%            │
echo   │  API docs:   http://127.0.0.1:%PORT%/docs       │
echo   │  Health:     http://127.0.0.1:%PORT%/api/v1/health │
echo   │                                              │
echo   │  Press Ctrl+C to stop                        │
echo   └──────────────────────────────────────────────┘
echo.
uvicorn main:app --host 127.0.0.1 --port %PORT% --reload
exit /b 0

:: ── Run tests ────────────────────────────────────────────────────────────────
:run_tests
echo [INFO] Running test suite...
python -m pytest tests/ -v
exit /b 0

:: ── Clean ────────────────────────────────────────────────────────────────────
:clean
echo [INFO] Cleaning up...
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
if exist "%SCRIPT_DIR%__pycache__" rmdir /s /q "%SCRIPT_DIR%__pycache__"
if exist "%SCRIPT_DIR%tests\__pycache__" rmdir /s /q "%SCRIPT_DIR%tests\__pycache__"
if exist "%SCRIPT_DIR%.pytest_cache" rmdir /s /q "%SCRIPT_DIR%.pytest_cache"
if exist "%SCRIPT_DIR%verification_plot.png" del /f "%SCRIPT_DIR%verification_plot.png"
if exist "%MODEL_DIR%" rmdir /s /q "%MODEL_DIR%"
echo [OK]   Cleaned venv, caches, models, and generated files
exit /b 0

:: ── Help ─────────────────────────────────────────────────────────────────────
:show_help
echo Usage: setup.bat [option]
echo.
echo Options:
echo   (none)       Full setup: create venv, install deps, train model, start server
echo   --serve, -s  Start the server (assumes deps + model exist)
echo   --train, -t  Train the model only
echo   --test       Run the test suite
echo   --clean, -c  Remove venv and cached files
echo   --help,  -h  Show this help
echo.
echo Environment variables:
echo   PORT         Server port (default: 8000)
exit /b 0
