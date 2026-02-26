"""Tests for the PDE equation registry."""

import pytest
import torch

from equations import (
    EQUATIONS,
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
    get_equation,
    list_equations,
)
from model import PhysicsInformedNN


class TestRegistry:
    """Tests for the equation registry."""

    def test_all_equations_registered(self):
        assert len(EQUATIONS) == 10

    def test_all_keys_are_strings(self):
        for key in EQUATIONS:
            assert isinstance(key, str)
            assert len(key) > 0

    def test_get_equation_valid(self):
        eq = get_equation("burgers")
        assert eq.key == "burgers"

    def test_get_equation_invalid(self):
        with pytest.raises(KeyError, match="Unknown equation"):
            get_equation("nonexistent")

    def test_list_equations_returns_all(self):
        result = list_equations()
        assert len(result) == 10
        assert all("key" in eq for eq in result)
        assert all("name" in eq for eq in result)
        assert all("latex_equation" in eq for eq in result)

    def test_all_equations_have_required_fields(self):
        for key, eq in EQUATIONS.items():
            assert eq.key == key
            assert eq.name
            assert eq.description
            assert eq.latex_equation
            assert eq.latex_ic
            assert eq.latex_bc
            assert eq.latex_domain
            assert eq.x_min < eq.x_max
            assert eq.t_min < eq.t_max
            assert eq.y_min < eq.y_max


class TestPhysicsLoss:
    """Test that all equations produce valid physics losses."""

    @pytest.fixture(params=list(EQUATIONS.keys()))
    def equation(self, request):
        return EQUATIONS[request.param]

    def test_physics_loss_is_scalar(self, equation):
        model = PhysicsInformedNN(hidden_size=8, num_layers=2)
        x = torch.rand(50, 1) * (equation.x_max - equation.x_min) + equation.x_min
        t = torch.rand(50, 1) * (equation.t_max - equation.t_min) + equation.t_min
        loss = equation.physics_loss(model, x, t)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_physics_loss_is_differentiable(self, equation):
        model = PhysicsInformedNN(hidden_size=8, num_layers=2)
        x = torch.rand(20, 1) * (equation.x_max - equation.x_min) + equation.x_min
        t = torch.rand(20, 1) * (equation.t_max - equation.t_min) + equation.t_min
        loss = equation.physics_loss(model, x, t)
        loss.backward()
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads


class TestBoundaryConditions:
    """Test IC/BC for all equations."""

    @pytest.fixture(params=list(EQUATIONS.keys()))
    def equation(self, request):
        return EQUATIONS[request.param]

    def test_initial_condition_shape(self, equation):
        x = torch.rand(100, 1) * (equation.x_max - equation.x_min) + equation.x_min
        u = equation.initial_condition(x)
        assert u.shape == (100, 1)
        assert torch.isfinite(u).all()

    def test_bc_left_shape(self, equation):
        t = torch.rand(50, 1)
        u = equation.bc_left(t)
        assert u.shape == (50, 1)
        assert torch.isfinite(u).all()

    def test_bc_right_shape(self, equation):
        t = torch.rand(50, 1)
        u = equation.bc_right(t)
        assert u.shape == (50, 1)
        assert torch.isfinite(u).all()


class TestSpecificEquations:
    """Spot-check specific equation properties."""

    def test_burgers_nu(self):
        eq = BurgersEquation()
        assert eq.params["nu"] == pytest.approx(0.01 / 3.14159265, rel=1e-4)

    def test_heat_alpha(self):
        eq = HeatEquation()
        assert eq.params["alpha"] == 0.05

    def test_wave_c(self):
        eq = WaveEquation()
        assert eq.params["c"] == 1.0

    def test_kdv_domain(self):
        eq = KdVEquation()
        assert eq.x_min == -5.0
        assert eq.x_max == 5.0

    def test_allen_cahn_bc(self):
        eq = AllenCahnEquation()
        t = torch.rand(10, 1)
        assert torch.allclose(eq.bc_left(t), -torch.ones(10, 1))
        assert torch.allclose(eq.bc_right(t), -torch.ones(10, 1))

    def test_reaction_diffusion_ic(self):
        eq = ReactionDiffusionEquation()
        x = torch.zeros(1, 1)
        u = eq.initial_condition(x)
        assert u.item() == pytest.approx(1.0)  # e^0 = 1
