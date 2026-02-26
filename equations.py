"""PDE equation registry — defines supported PDEs with physics losses, ICs/BCs, and metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx via autograd, keeping the graph alive."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


@dataclass
class PDEEquation:
    """Definition of a PDE problem for PINN solving."""

    # Identity
    key: str                         # unique slug, e.g. "burgers"
    name: str                        # display name
    description: str                 # one-liner for the dashboard

    # LaTeX strings for the info cards
    latex_equation: str
    latex_ic: str
    latex_bc: str
    latex_domain: str

    # Domain
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # Plot config
    y_min: float = -1.3
    y_max: float = 1.3
    y_label: str = "u(x, t)"
    plot_title_template: str = "Solution at t = {t:.2f}"

    # Training hints (can be overridden via CLI)
    recommended_epochs: int = 10_000
    recommended_hidden: int = 20
    recommended_layers: int = 4
    recommended_n_f: int = 10_000
    recommended_n_bc: int = 500

    # Extra physics parameters stored as dict
    params: dict[str, float] = field(default_factory=dict)

    # Callables — set per-equation below
    # physics_loss_fn(model, x, t, params) -> scalar loss
    # ic_fn(x, params) -> u values at t=0
    # bc_left_fn(t, params) -> u values at x=x_min
    # bc_right_fn(t, params) -> u values at x=x_max

    def physics_loss(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def bc_left(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def bc_right(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# --- Burgers: u_t + u * u_x = nu * u_xx
class BurgersEquation(PDEEquation):
    def __init__(self, nu: float = 0.01 / math.pi):
        super().__init__(
            key="burgers",
            name="Burgers Equation",
            description="Viscous Burgers equation — nonlinear advection-diffusion",
            latex_equation=r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}",
            latex_ic=r"u(x, 0) = -\sin(\pi x)",
            latex_bc=r"u(-1, t) = u(1, t) = 0",
            latex_domain=r"x \in [-1, 1], \; t \in [0, 1]",
            y_min=-1.3, y_max=1.3,
            y_label="u(x, t)",
            params={"nu": nu},
            recommended_epochs=10_000,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_x = _grad(u, x)
        u_t = _grad(u, t)
        u_xx = _grad(u_x, x)
        nu = self.params["nu"]
        f = u_t + u * u_x - nu * u_xx
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return -torch.sin(torch.pi * x)

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Heat: u_t = alpha * u_xx
class HeatEquation(PDEEquation):
    def __init__(self, alpha: float = 0.05):
        super().__init__(
            key="heat",
            name="Heat Equation",
            description="Linear diffusion equation — thermal conduction",
            latex_equation=r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}",
            latex_ic=r"u(x, 0) = \sin(\pi x)",
            latex_bc=r"u(-1, t) = u(1, t) = 0",
            latex_domain=r"x \in [-1, 1], \; t \in [0, 1]",
            y_min=-1.2, y_max=1.2,
            y_label="Temperature u(x, t)",
            params={"alpha": alpha},
            recommended_epochs=5_000,
            recommended_hidden=20,
            recommended_layers=4,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_t = _grad(u, t)
        u_x = _grad(u, x)
        u_xx = _grad(u_x, x)
        alpha = self.params["alpha"]
        f = u_t - alpha * u_xx
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return torch.sin(torch.pi * x)

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Wave: u_tt = c^2 * u_xx (second-order, network outputs u directly)
class WaveEquation(PDEEquation):
    def __init__(self, c: float = 1.0):
        super().__init__(
            key="wave",
            name="Wave Equation",
            description="Linear wave equation — propagation of disturbances",
            latex_equation=r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}",
            latex_ic=r"u(x,0) = \sin(\pi x), \; u_t(x,0) = 0",
            latex_bc=r"u(-1, t) = u(1, t) = 0",
            latex_domain=r"x \in [-1, 1], \; t \in [0, 1]",
            y_min=-1.3, y_max=1.3,
            y_label="Displacement u(x, t)",
            params={"c": c},
            recommended_epochs=15_000,
            recommended_hidden=32,
            recommended_layers=5,
            recommended_n_f=15_000,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_x = _grad(u, x)
        u_t = _grad(u, t)
        u_xx = _grad(u_x, x)
        u_tt = _grad(u_t, t)
        c = self.params["c"]
        f = u_tt - c**2 * u_xx
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return torch.sin(torch.pi * x)

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Advection: u_t + c * u_x = 0
class AdvectionEquation(PDEEquation):
    def __init__(self, c: float = 1.0):
        super().__init__(
            key="advection",
            name="Advection Equation",
            description="Linear advection — transport of a profile at constant speed",
            latex_equation=r"\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0",
            latex_ic=r"u(x, 0) = -\sin(\pi x)",
            latex_bc=r"u(-1, t) = u(1, t) = 0",
            latex_domain=r"x \in [-1, 1], \; t \in [0, 1]",
            y_min=-1.3, y_max=1.3,
            y_label="u(x, t)",
            params={"c": c},
            recommended_epochs=8_000,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_x = _grad(u, x)
        u_t = _grad(u, t)
        c = self.params["c"]
        f = u_t + c * u_x
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return -torch.sin(torch.pi * x)

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Convection-Diffusion: u_t + c * u_x = nu * u_xx
class ConvectionDiffusionEquation(PDEEquation):
    def __init__(self, c: float = 1.0, nu: float = 0.05):
        super().__init__(
            key="convection_diffusion",
            name="Convection-Diffusion",
            description="Linear convection-diffusion — advection with viscous damping",
            latex_equation=r"\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}",
            latex_ic=r"u(x, 0) = -\sin(\pi x)",
            latex_bc=r"u(-1, t) = u(1, t) = 0",
            latex_domain=r"x \in [-1, 1], \; t \in [0, 1]",
            y_min=-1.3, y_max=1.3,
            y_label="u(x, t)",
            params={"c": c, "nu": nu},
            recommended_epochs=8_000,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_x = _grad(u, x)
        u_t = _grad(u, t)
        u_xx = _grad(u_x, x)
        c = self.params["c"]
        nu = self.params["nu"]
        f = u_t + c * u_x - nu * u_xx
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return -torch.sin(torch.pi * x)

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Reaction-Diffusion (Fisher-KPP): u_t = D * u_xx + r * u * (1 - u)
class ReactionDiffusionEquation(PDEEquation):
    def __init__(self, D: float = 0.01, r: float = 1.0):
        super().__init__(
            key="reaction_diffusion",
            name="Reaction-Diffusion (Fisher-KPP)",
            description="Fisher equation — diffusion with logistic growth",
            latex_equation=r"\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + r \, u (1 - u)",
            latex_ic=r"u(x, 0) = e^{-10 x^2}",
            latex_bc=r"u(-1, t) = 0, \; u(1, t) = 0",
            latex_domain=r"x \in [-1, 1], \; t \in [0, 1]",
            x_min=-1.0, x_max=1.0,
            y_min=-0.2, y_max=1.3,
            y_label="Concentration u(x, t)",
            params={"D": D, "r": r},
            recommended_epochs=15_000,
            recommended_hidden=32,
            recommended_layers=5,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_t = _grad(u, t)
        u_x = _grad(u, x)
        u_xx = _grad(u_x, x)
        D = self.params["D"]
        r = self.params["r"]
        f = u_t - D * u_xx - r * u * (1 - u)
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return torch.exp(-10.0 * x ** 2)

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Allen-Cahn: u_t = eps^2 * u_xx + u - u^3
class AllenCahnEquation(PDEEquation):
    def __init__(self, eps: float = 0.1):
        super().__init__(
            key="allen_cahn",
            name="Allen-Cahn Equation",
            description="Phase-field model — interface dynamics and phase separation",
            latex_equation=r"\frac{\partial u}{\partial t} = \varepsilon^2 \frac{\partial^2 u}{\partial x^2} + u - u^3",
            latex_ic=r"u(x, 0) = x^2 \cos(\pi x)",
            latex_bc=r"u(-1, t) = u(1, t) = -1",
            latex_domain=r"x \in [-1, 1], \; t \in [0, 1]",
            y_min=-1.5, y_max=1.5,
            y_label="Phase u(x, t)",
            params={"eps": eps},
            recommended_epochs=20_000,
            recommended_hidden=32,
            recommended_layers=5,
            recommended_n_f=15_000,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_t = _grad(u, t)
        u_x = _grad(u, x)
        u_xx = _grad(u_x, x)
        eps = self.params["eps"]
        f = u_t - eps**2 * u_xx - u + u**3
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return x**2 * torch.cos(torch.pi * x)

    def bc_left(self, t):
        return -torch.ones_like(t)

    def bc_right(self, t):
        return -torch.ones_like(t)


# --- KdV: u_t + 6*u*u_x + u_xxx = 0
class KdVEquation(PDEEquation):
    def __init__(self):
        super().__init__(
            key="kdv",
            name="Korteweg-de Vries (KdV)",
            description="Soliton equation — nonlinear dispersive waves",
            latex_equation=r"\frac{\partial u}{\partial t} + 6 u \frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0",
            latex_ic=r"u(x, 0) = -2 \, \mathrm{sech}^2(x)",
            latex_bc=r"u(\pm 5, t) = 0",
            latex_domain=r"x \in [-5, 5], \; t \in [0, 1]",
            x_min=-5.0, x_max=5.0,
            y_min=-2.5, y_max=0.5,
            y_label="u(x, t)",
            recommended_epochs=20_000,
            recommended_hidden=40,
            recommended_layers=5,
            recommended_n_f=15_000,
            recommended_n_bc=800,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_x = _grad(u, x)
        u_t = _grad(u, t)
        u_xx = _grad(u_x, x)
        u_xxx = _grad(u_xx, x)
        f = u_t + 6 * u * u_x + u_xxx
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return -2.0 / torch.cosh(x) ** 2

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Burgers (shock): u_t + u * u_x = nu * u_xx, nu = 0.001/pi
class BurgersShockEquation(PDEEquation):
    def __init__(self, nu: float = 0.001 / math.pi):
        super().__init__(
            key="burgers_shock",
            name="Burgers Equation (Shock)",
            description="Burgers equation at high Reynolds number — sharp shock formation",
            latex_equation=r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \quad \nu = 0.001/\pi",
            latex_ic=r"u(x, 0) = -\sin(\pi x)",
            latex_bc=r"u(-1, t) = u(1, t) = 0",
            latex_domain=r"x \in [-1, 1], \; t \in [0, 1]",
            y_min=-1.3, y_max=1.3,
            y_label="u(x, t)",
            params={"nu": nu},
            recommended_epochs=20_000,
            recommended_hidden=40,
            recommended_layers=6,
            recommended_n_f=20_000,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_x = _grad(u, x)
        u_t = _grad(u, t)
        u_xx = _grad(u_x, x)
        nu = self.params["nu"]
        f = u_t + u * u_x - nu * u_xx
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return -torch.sin(torch.pi * x)

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Cubic NLS-like: u_t + u_xx + u^3 = 0
class CubicNLSEquation(PDEEquation):
    def __init__(self):
        super().__init__(
            key="cubic_nls",
            name="Cubic Nonlinear PDE",
            description="Nonlinear dispersive equation — u_t + u_xx + u³ = 0",
            latex_equation=r"\frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} + u^3 = 0",
            latex_ic=r"u(x, 0) = \mathrm{sech}(x)",
            latex_bc=r"u(\pm 5, t) = 0",
            latex_domain=r"x \in [-5, 5], \; t \in [0, 1]",
            x_min=-5.0, x_max=5.0,
            y_min=-0.5, y_max=1.5,
            y_label="u(x, t)",
            recommended_epochs=15_000,
            recommended_hidden=32,
            recommended_layers=5,
            recommended_n_f=12_000,
        )

    def physics_loss(self, model, x, t):
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = model(x, t)
        u_t = _grad(u, t)
        u_x = _grad(u, x)
        u_xx = _grad(u_x, x)
        f = u_t + u_xx + u ** 3
        return torch.mean(f ** 2)

    def initial_condition(self, x):
        return 1.0 / torch.cosh(x)

    def bc_left(self, t):
        return torch.zeros_like(t)

    def bc_right(self, t):
        return torch.zeros_like(t)


# --- Registry
EQUATIONS: dict[str, PDEEquation] = {}


def _register(*eq_classes):
    for cls in eq_classes:
        eq = cls()
        EQUATIONS[eq.key] = eq


_register(
    BurgersEquation,
    HeatEquation,
    WaveEquation,
    AdvectionEquation,
    ConvectionDiffusionEquation,
    ReactionDiffusionEquation,
    AllenCahnEquation,
    KdVEquation,
    BurgersShockEquation,
    CubicNLSEquation,
)


def get_equation(key: str) -> PDEEquation:
    """Look up an equation by key. Raises KeyError if not found."""
    if key not in EQUATIONS:
        available = ", ".join(sorted(EQUATIONS.keys()))
        raise KeyError(f"Unknown equation '{key}'. Available: {available}")
    return EQUATIONS[key]


def list_equations() -> list[dict]:
    """Return a list of equation metadata dicts (for the API)."""
    return [
        {
            "key": eq.key,
            "name": eq.name,
            "description": eq.description,
            "latex_equation": eq.latex_equation,
            "latex_ic": eq.latex_ic,
            "latex_bc": eq.latex_bc,
            "latex_domain": eq.latex_domain,
            "x_min": eq.x_min,
            "x_max": eq.x_max,
            "t_min": eq.t_min,
            "t_max": eq.t_max,
            "y_min": eq.y_min,
            "y_max": eq.y_max,
            "y_label": eq.y_label,
            "plot_title_template": eq.plot_title_template,
            "params": eq.params,
        }
        for eq in EQUATIONS.values()
    ]
