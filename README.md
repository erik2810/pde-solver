# pde-solver

PINN-based solver for 10 partial differential equations, with a training CLI and interactive browser dashboard.

## What is this

This project uses [Physics-Informed Neural Networks](https://maziarraissi.github.io/PINNs/) (PINNs) to solve a collection of classical PDEs. Each equation gets its own trained model, served through a FastAPI backend and visualized in a single-page dashboard where you can scrub through time, switch equations, and export data.

There's also a [live demo](https://YOUR_USERNAME.github.io/pde-solver/) on GitHub Pages that runs inference entirely client-side.

![demo](docs/screenshot.png)

## Quick start

The setup script creates a venv, installs deps, trains all 10 models, and starts the server:

```bash
chmod +x setup.sh && ./setup.sh
```

Or do it manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python pinn_model.py --equation all --no-show-plot
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000.

## Supported equations

- **Burgers** -- viscous nonlinear advection-diffusion
- **Heat** -- linear diffusion / thermal conduction
- **Wave** -- linear wave propagation
- **Advection** -- transport at constant speed
- **Convection-Diffusion** -- advection with viscous damping
- **Reaction-Diffusion** (Fisher-KPP) -- diffusion with logistic growth
- **Allen-Cahn** -- phase-field interface dynamics
- **KdV** -- Korteweg-de Vries soliton equation
- **Burgers (Shock)** -- high Reynolds number, sharp shock formation
- **Cubic NLS** -- nonlinear dispersive equation (u_t + u_xx + u^3 = 0)

## Training

```bash
# train a single equation
python pinn_model.py --equation heat

# train everything
python pinn_model.py --equation all --no-show-plot

# see what's available
python pinn_model.py --list
```

Each equation ships with its own default hyperparameters. You can override them:

| Flag | Description |
|------|-------------|
| `--equation` | Equation key or `all` (default: `burgers`) |
| `--epochs` | Number of training epochs |
| `--lr` | Learning rate (default: 0.001) |
| `--no-show-plot` | Don't open the matplotlib window |
| `--save-plot` | Save verification plot to disk |

Trained weights go to `models/{equation_key}_model.pth`.

## API

The server (`main.py`) exposes a few endpoints:

- `GET /api/v1/health` -- status, loaded models, uptime
- `GET /api/v1/equations` -- metadata for all registered equations
- `POST /api/v1/predict/{equation_key}` -- predict u(x) at a given time `t`

The predict endpoint takes `{"t": 0.5, "n_points": 100}` and returns arrays of `x` and `u` values.

## Adding new equations

- Subclass `PDEEquation` in `equations.py`
- Implement `physics_loss()`, `initial_condition()`, `bc_left()`, `bc_right()`
- Register it via `_register(YourClass)` at the bottom of the file
- Train with `python pinn_model.py --equation your_key` -- the dashboard picks it up automatically

## Live demo

The `docs/` directory contains a static build of the dashboard that runs on [GitHub Pages](https://YOUR_USERNAME.github.io/pde-solver/). It loads pre-exported model weights and does inference in JavaScript, so no backend needed.

## License

MIT
