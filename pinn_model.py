"""PINN training script for all registered PDEs.

Usage:
    python pinn_model.py                    # Burgers (default)
    python pinn_model.py --equation heat
    python pinn_model.py --equation all
    python pinn_model.py --list
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from equations import EQUATIONS, PDEEquation, get_equation, list_equations
from model import PhysicsInformedNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_EPOCHS: int = 10_000
DEFAULT_LR: float = 1e-3
DEFAULT_N_F: int = 10_000
DEFAULT_N_BC: int = 500
DEFAULT_HIDDEN_SIZE: int = 20
DEFAULT_NUM_LAYERS: int = 4
DEFAULT_SEED: int = 42
DEFAULT_PATIENCE: int = 1000
DEFAULT_PHYSICS_WEIGHT: float = 1.0
DEFAULT_DATA_WEIGHT: float = 1.0
DEFAULT_OUTPUT_DIR: str = "models"


def get_training_data(
    equation: PDEEquation,
    n_f: int = DEFAULT_N_F,
    n_bc: int = DEFAULT_N_BC,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate collocation points and BC/IC data. Returns (x_f, t_f, x_bc, t_bc, u_bc)."""
    device = device or torch.device("cpu")
    x_range = equation.x_max - equation.x_min
    t_range = equation.t_max - equation.t_min

    # Interior collocation points
    x_f = (torch.rand(n_f, 1) * x_range + equation.x_min).to(device)
    t_f = (torch.rand(n_f, 1) * t_range + equation.t_min).to(device)

    # Initial condition at t = t_min
    x_ic = (torch.rand(n_bc, 1) * x_range + equation.x_min).to(device)
    t_ic = torch.full((n_bc, 1), equation.t_min, device=device)
    u_ic = equation.initial_condition(x_ic)

    t_bc_rand = (torch.rand(n_bc, 1) * t_range + equation.t_min).to(device)

    # Left boundary
    x_bc_left = torch.full((n_bc, 1), equation.x_min, device=device)
    u_bc_left = equation.bc_left(t_bc_rand)

    # Right boundary
    x_bc_right = torch.full((n_bc, 1), equation.x_max, device=device)
    u_bc_right = equation.bc_right(t_bc_rand)

    x_bc = torch.cat([x_bc_left, x_bc_right, x_ic], dim=0)
    t_bc = torch.cat([t_bc_rand, t_bc_rand, t_ic], dim=0)
    u_bc = torch.cat([u_bc_left, u_bc_right, u_ic], dim=0)

    return x_f, t_f, x_bc, t_bc, u_bc


def train(
    equation: PDEEquation | str = "burgers",
    epochs: int | None = None,
    lr: float = DEFAULT_LR,
    n_f: int | None = None,
    n_bc: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    physics_weight: float = DEFAULT_PHYSICS_WEIGHT,
    data_weight: float = DEFAULT_DATA_WEIGHT,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
    output_path: str | None = None,
    save_plot: bool = False,
    show_plot: bool = True,
) -> PhysicsInformedNN:
    """Train a PINN for the given equation. None-valued params fall back to equation defaults."""
    if isinstance(equation, str):
        equation = get_equation(equation)

    epochs = epochs if epochs is not None else equation.recommended_epochs
    n_f = n_f if n_f is not None else equation.recommended_n_f
    n_bc = n_bc if n_bc is not None else equation.recommended_n_bc
    hidden_size = hidden_size if hidden_size is not None else equation.recommended_hidden
    num_layers = num_layers if num_layers is not None else equation.recommended_layers

    if output_path is None:
        Path(DEFAULT_OUTPUT_DIR).mkdir(exist_ok=True)
        output_path = f"{DEFAULT_OUTPUT_DIR}/{equation.key}_model.pth"

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Equation: %s (%s)", equation.name, equation.key)
    logger.info("Device: %s | Epochs: %d | LR: %.1e", device, epochs, lr)
    logger.info("Architecture: %d layers × %d hidden", num_layers, hidden_size)

    model = PhysicsInformedNN(hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 4, min_lr=1e-6
    )

    x_f, t_f, x_bc, t_bc, u_bc = get_training_data(equation, n_f, n_bc, device)

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_physics = equation.physics_loss(model, x_f, t_f)
        u_pred_bc = model(x_bc, t_bc)
        loss_data = torch.mean((u_pred_bc - u_bc) ** 2)
        loss = physics_weight * loss_physics + data_weight * loss_data

        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())

        current_loss = loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 500 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %5d | Loss: %.6f (Physics: %.6f, Data: %.6f) | LR: %.1e",
                epoch, current_loss, loss_physics.item(), loss_data.item(), current_lr,
            )

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    elapsed = time.time() - start_time
    logger.info("Training finished in %.1f s — best loss: %.6f", elapsed, best_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info("Model saved to '%s'", output_path)

    _plot_verification(model, equation, device, save_plot, show_plot)
    return model


def _plot_verification(
    model: PhysicsInformedNN,
    equation: PDEEquation,
    device: torch.device,
    save: bool = False,
    show: bool = True,
) -> None:
    if not save and not show:
        return

    if save and not show:
        matplotlib.use("Agg")

    model.eval()
    test_x = torch.linspace(equation.x_min, equation.x_max, 200, device=device).view(-1, 1)
    t_range = equation.t_max - equation.t_min
    time_steps = [equation.t_min + t_range * f for f in [0.0, 0.25, 0.5, 0.75, 1.0]]

    plt.figure(figsize=(10, 6))
    with torch.no_grad():
        for t_val in time_steps:
            test_t = torch.full_like(test_x, t_val)
            prediction = model(test_x, test_t)
            plt.plot(
                test_x.cpu().numpy().flatten(),
                prediction.cpu().numpy().flatten(),
                label=f"t = {t_val:.2f}",
            )

    plt.title(f"{equation.name} — PINN Solution")
    plt.xlabel("x")
    plt.ylabel(equation.y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        fname = f"verification_{equation.key}.png"
        plt.savefig(fname, dpi=150)
        logger.info("Plot saved to '%s'", fname)
    if show:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    eq_keys = ", ".join(sorted(EQUATIONS.keys()))
    parser = argparse.ArgumentParser(
        description="Train a PINN to solve a PDE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available equations: {eq_keys}\n\nUse --equation all to train all equations.",
    )
    parser.add_argument(
        "--equation", type=str, default="burgers",
        help=f"Equation to train (or 'all'). Choices: {eq_keys}",
    )
    parser.add_argument("--list", action="store_true", help="List available equations and exit")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: equation-specific)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate")
    parser.add_argument("--n-f", type=int, default=None, help="Number of collocation points")
    parser.add_argument("--n-bc", type=int, default=None, help="Number of BC/IC points")
    parser.add_argument("--hidden-size", type=int, default=None, help="Neurons per hidden layer")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of hidden layers")
    parser.add_argument("--physics-weight", type=float, default=DEFAULT_PHYSICS_WEIGHT)
    parser.add_argument("--data-weight", type=float, default=DEFAULT_DATA_WEIGHT)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    parser.add_argument("--save-plot", action="store_true")
    parser.add_argument("--no-show-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print("\nAvailable equations:\n")
        for eq_info in list_equations():
            print(f"  {eq_info['key']:<25s} {eq_info['name']}")
            print(f"  {'':<25s} {eq_info['description']}")
            print()
        raise SystemExit(0)

    equations_to_train: list[str]
    if args.equation == "all":
        equations_to_train = list(EQUATIONS.keys())
    else:
        equations_to_train = [args.equation]

    for eq_key in equations_to_train:
        print(f"\n{'='*60}")
        print(f"  Training: {eq_key}")
        print(f"{'='*60}\n")
        train(
            equation=eq_key,
            epochs=args.epochs,
            lr=args.lr,
            n_f=args.n_f,
            n_bc=args.n_bc,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            physics_weight=args.physics_weight,
            data_weight=args.data_weight,
            patience=args.patience,
            seed=args.seed,
            output_path=args.output if len(equations_to_train) == 1 else None,
            save_plot=args.save_plot,
            show_plot=not args.no_show_plot,
        )
