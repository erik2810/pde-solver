"""Tests for the PINN training pipeline."""

import pytest
import torch

from equations import get_equation, BurgersEquation
from pinn_model import get_training_data, train


@pytest.fixture
def burgers():
    return get_equation("burgers")


@pytest.fixture
def heat():
    return get_equation("heat")


class TestGetTrainingData:
    """Tests for training data generation."""

    def test_default_sizes(self, burgers):
        x_f, t_f, x_bc, t_bc, u_bc = get_training_data(burgers)
        n_f = burgers.recommended_n_f
        n_bc = burgers.recommended_n_bc
        assert x_f.shape == (n_f, 1)
        assert t_f.shape == (n_f, 1)
        # BC: n_bc left + n_bc right + n_bc IC = 3 * n_bc
        assert x_bc.shape == (3 * n_bc, 1)
        assert t_bc.shape == (3 * n_bc, 1)
        assert u_bc.shape == (3 * n_bc, 1)

    def test_custom_sizes(self, burgers):
        x_f, t_f, x_bc, t_bc, u_bc = get_training_data(burgers, n_f=100, n_bc=50)
        assert x_f.shape == (100, 1)
        assert t_f.shape == (100, 1)
        assert x_bc.shape == (150, 1)  # 3 * 50

    def test_collocation_points_in_domain(self, burgers):
        x_f, t_f, _, _, _ = get_training_data(burgers, n_f=5000, n_bc=50)
        assert x_f.min() >= burgers.x_min
        assert x_f.max() <= burgers.x_max
        assert t_f.min() >= burgers.t_min
        assert t_f.max() <= burgers.t_max

    def test_collocation_points_in_domain_kdv(self):
        """KdV has a wider domain [-5, 5]; verify data respects it."""
        kdv = get_equation("kdv")
        x_f, t_f, _, _, _ = get_training_data(kdv, n_f=5000, n_bc=50)
        assert x_f.min() >= kdv.x_min
        assert x_f.max() <= kdv.x_max

    def test_initial_condition_values(self, burgers):
        """IC points should have t=0 and u = -sin(pi * x)."""
        _, _, x_bc, t_bc, u_bc = get_training_data(burgers, n_f=100, n_bc=200)
        # Last 200 entries are IC
        t_ic = t_bc[-200:]
        x_ic = x_bc[-200:]
        u_ic = u_bc[-200:]
        assert torch.allclose(t_ic, torch.zeros_like(t_ic))
        expected_u = -torch.sin(torch.pi * x_ic)
        assert torch.allclose(u_ic, expected_u)

    def test_boundary_condition_values(self, burgers):
        """BC points at x=-1 and x=1 should have u=0 for Burgers."""
        _, _, x_bc, _, u_bc = get_training_data(burgers, n_f=100, n_bc=100)
        # First 100: left boundary (x=-1), next 100: right boundary (x=1)
        assert torch.allclose(x_bc[:100], torch.full((100, 1), burgers.x_min))
        assert torch.allclose(x_bc[100:200], torch.full((100, 1), burgers.x_max))
        assert torch.allclose(u_bc[:200], torch.zeros(200, 1))

    def test_allen_cahn_bc_values(self):
        """Allen-Cahn has bc_left = bc_right = -1."""
        eq = get_equation("allen_cahn")
        _, _, x_bc, t_bc, u_bc = get_training_data(eq, n_f=100, n_bc=50)
        # First 50: left BC, next 50: right BC
        assert torch.allclose(u_bc[:50], -torch.ones(50, 1))
        assert torch.allclose(u_bc[50:100], -torch.ones(50, 1))

    def test_device_placement(self, burgers):
        device = torch.device("cpu")
        x_f, t_f, x_bc, t_bc, u_bc = get_training_data(
            burgers, n_f=10, n_bc=5, device=device
        )
        for tensor in [x_f, t_f, x_bc, t_bc, u_bc]:
            assert tensor.device.type == "cpu"

    def test_reproducibility_with_seed(self, burgers):
        torch.manual_seed(123)
        data1 = get_training_data(burgers, n_f=100, n_bc=50)
        torch.manual_seed(123)
        data2 = get_training_data(burgers, n_f=100, n_bc=50)
        for t1, t2 in zip(data1, data2):
            assert torch.allclose(t1, t2)

    def test_all_equations_produce_valid_data(self):
        """Every registered equation should produce valid training data."""
        from equations import EQUATIONS

        for key, eq in EQUATIONS.items():
            x_f, t_f, x_bc, t_bc, u_bc = get_training_data(eq, n_f=50, n_bc=20)
            assert x_f.shape == (50, 1), f"Failed for {key}"
            assert torch.isfinite(x_bc).all(), f"Non-finite BC x for {key}"
            assert torch.isfinite(u_bc).all(), f"Non-finite BC u for {key}"


class TestTrain:
    """Tests for the training function (short runs)."""

    def test_quick_train_completes(self, tmp_path):
        output = tmp_path / "test_model.pth"
        model = train(
            equation="burgers",
            epochs=10,
            n_f=100,
            n_bc=50,
            hidden_size=8,
            num_layers=2,
            patience=100,
            output_path=str(output),
            show_plot=False,
            save_plot=False,
        )
        assert output.exists()
        assert model is not None

    def test_train_heat_equation(self, tmp_path):
        output = tmp_path / "heat_model.pth"
        model = train(
            equation="heat",
            epochs=10,
            n_f=100,
            n_bc=50,
            hidden_size=8,
            num_layers=2,
            patience=100,
            output_path=str(output),
            show_plot=False,
            save_plot=False,
        )
        assert output.exists()
        assert model is not None

    def test_train_with_equation_object(self, tmp_path):
        eq = BurgersEquation()
        output = tmp_path / "test_model.pth"
        model = train(
            equation=eq,
            epochs=10,
            n_f=100,
            n_bc=50,
            hidden_size=8,
            num_layers=2,
            patience=100,
            output_path=str(output),
            show_plot=False,
            save_plot=False,
        )
        assert output.exists()
        assert model is not None

    def test_model_produces_predictions_after_training(self, tmp_path):
        output = tmp_path / "test_model.pth"
        model = train(
            equation="burgers",
            epochs=50,
            n_f=200,
            n_bc=50,
            hidden_size=8,
            num_layers=2,
            patience=100,
            output_path=str(output),
            show_plot=False,
            save_plot=False,
        )
        model = model.cpu()
        model.eval()
        x = torch.linspace(-1, 1, 50).view(-1, 1)
        t = torch.zeros(50, 1)
        with torch.no_grad():
            pred = model(x, t)
        assert pred.shape == (50, 1)
        assert torch.isfinite(pred).all()

    def test_early_stopping_triggers(self, tmp_path):
        """With very short patience and many epochs, early stopping should trigger."""
        output = tmp_path / "test_model.pth"
        model = train(
            equation="burgers",
            epochs=100_000,
            n_f=50,
            n_bc=20,
            hidden_size=4,
            num_layers=2,
            patience=5,
            output_path=str(output),
            show_plot=False,
            save_plot=False,
        )
        assert model is not None

    def test_saved_model_is_loadable(self, tmp_path):
        from model import PhysicsInformedNN

        output = tmp_path / "test_model.pth"
        train(
            equation="burgers",
            epochs=10,
            n_f=50,
            n_bc=20,
            hidden_size=8,
            num_layers=2,
            output_path=str(output),
            show_plot=False,
            save_plot=False,
        )

        loaded = PhysicsInformedNN(hidden_size=8, num_layers=2)
        loaded.load_state_dict(torch.load(str(output), weights_only=True))
        loaded.eval()

        x = torch.tensor([[0.0]])
        t = torch.tensor([[0.5]])
        with torch.no_grad():
            out = loaded(x, t)
        assert torch.isfinite(out).all()

    def test_reproducibility(self, tmp_path):
        """Same seed should produce same results."""
        kwargs = dict(
            equation="burgers",
            epochs=20,
            n_f=100,
            n_bc=50,
            hidden_size=8,
            num_layers=2,
            seed=999,
            patience=100,
            show_plot=False,
            save_plot=False,
        )

        out1 = tmp_path / "m1.pth"
        out2 = tmp_path / "m2.pth"

        model1 = train(**kwargs, output_path=str(out1))
        model2 = train(**kwargs, output_path=str(out2))

        model1 = model1.cpu()
        model2 = model2.cpu()
        x = torch.tensor([[0.0], [0.5], [-0.5]])
        t = torch.tensor([[0.0], [0.5], [1.0]])
        with torch.no_grad():
            p1 = model1(x, t)
            p2 = model2(x, t)
        assert torch.allclose(p1, p2, atol=1e-6)

    def test_default_output_path(self, tmp_path, monkeypatch):
        """When output_path is None, model is saved to models/{key}_model.pth."""
        models_dir = tmp_path / "models"
        monkeypatch.setattr("pinn_model.DEFAULT_OUTPUT_DIR", str(models_dir))
        model = train(
            equation="burgers",
            epochs=5,
            n_f=50,
            n_bc=20,
            hidden_size=8,
            num_layers=2,
            patience=100,
            output_path=None,
            show_plot=False,
            save_plot=False,
        )
        assert (models_dir / "burgers_model.pth").exists()
