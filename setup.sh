#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PINN PDE Solver — macOS / Linux Setup & Run
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # full setup: venv, deps, train, serve
#   ./setup.sh --serve      # skip training, just start the server
#   ./setup.sh --train      # only train the model (no server)
#   ./setup.sh --test       # run the test suite
#   ./setup.sh --clean      # remove venv and cached files
#
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_DIR="$SCRIPT_DIR/models"
MODEL_FILE="$MODEL_DIR/burgers_model.pth"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
PORT="${PORT:-8000}"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*" >&2; }

# ── Python detection ──────────────────────────────────────────────────────────
find_python() {
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            local major minor
            major="${ver%%.*}"
            minor="${ver#*.}"
            if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# ── Virtual environment ───────────────────────────────────────────────────────
ensure_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating virtual environment in ${VENV_DIR/#$HOME/\~}..."
        "$PYTHON" -m venv "$VENV_DIR"
        ok "Virtual environment created"
    fi

    # Activate
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ok "Activated venv ($(python --version))"
}

# ── Dependencies ──────────────────────────────────────────────────────────────
install_deps() {
    info "Installing dependencies..."
    pip install --upgrade pip --quiet
    pip install -r "$REQUIREMENTS" --quiet
    ok "All dependencies installed"
}

# ── Train ─────────────────────────────────────────────────────────────────────
train_model() {
    local eq="${1:-all}"
    if [[ "$eq" == "all" && -d "$MODEL_DIR" ]] && ls "$MODEL_DIR"/*.pth &>/dev/null; then
        warn "Trained models already exist in models/"
        read -rp "    Retrain all from scratch? [y/N] " answer
        if [[ ! "$answer" =~ ^[Yy]$ ]]; then
            info "Skipping training"
            return 0
        fi
    fi

    info "Training PDE models (equation: $eq) — this may take several minutes..."
    python "$SCRIPT_DIR/pinn_model.py" --equation "$eq" --no-show-plot --save-plot
    ok "Training complete — models saved to models/"
}

# ── Serve ─────────────────────────────────────────────────────────────────────
start_server() {
    if [[ ! -d "$MODEL_DIR" ]] || ! ls "$MODEL_DIR"/*.pth &>/dev/null; then
        err "No trained models found. Run training first: ./setup.sh --train"
        exit 1
    fi

    info "Starting server on http://127.0.0.1:${PORT}"
    echo -e "${BOLD}"
    echo "  ┌──────────────────────────────────────────────┐"
    echo "  │  Dashboard:  http://127.0.0.1:${PORT}            │"
    echo "  │  API docs:   http://127.0.0.1:${PORT}/docs       │"
    echo "  │  Health:     http://127.0.0.1:${PORT}/api/v1/health │"
    echo "  │                                              │"
    echo "  │  Press Ctrl+C to stop                        │"
    echo "  └──────────────────────────────────────────────┘"
    echo -e "${NC}"

    uvicorn main:app --host 127.0.0.1 --port "$PORT" --reload
}

# ── Test ──────────────────────────────────────────────────────────────────────
run_tests() {
    info "Running test suite..."
    python -m pytest tests/ -v
}

# ── Clean ─────────────────────────────────────────────────────────────────────
clean() {
    info "Cleaning up..."
    rm -rf "$VENV_DIR"
    rm -rf "$SCRIPT_DIR/__pycache__" "$SCRIPT_DIR/tests/__pycache__"
    rm -rf "$SCRIPT_DIR/.pytest_cache"
    rm -f  "$SCRIPT_DIR/verification_plot.png"
    ok "Cleaned venv, caches, and generated files"
    rm -rf "$MODEL_DIR"
    ok "Also removed models/"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  PINN PDE Solver — Setup                     ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    # Check Python
    PYTHON="$(find_python)" || {
        err "Python 3.10+ is required but was not found."
        err "Install it from https://www.python.org/downloads/"
        exit 1
    }
    ok "Found $($PYTHON --version) at $(command -v "$PYTHON")"

    local mode="${1:-full}"

    case "$mode" in
        --serve|-s)
            ensure_venv
            start_server
            ;;
        --train|-t)
            ensure_venv
            install_deps
            train_model
            ;;
        --test)
            ensure_venv
            install_deps
            run_tests
            ;;
        --clean|-c)
            clean
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [option]"
            echo ""
            echo "Options:"
            echo "  (none)       Full setup: create venv, install deps, train model, start server"
            echo "  --serve, -s  Start the server (assumes deps + model exist)"
            echo "  --train, -t  Train the model only"
            echo "  --test       Run the test suite"
            echo "  --clean, -c  Remove venv and cached files"
            echo "  --help,  -h  Show this help"
            echo ""
            echo "Environment variables:"
            echo "  PORT         Server port (default: 8000)"
            ;;
        *)
            ensure_venv
            install_deps
            train_model
            start_server
            ;;
    esac
}

main "$@"
