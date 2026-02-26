"""Tests for the shared PINN model architecture."""

import pytest
import torch

from model import PhysicsInformedNN


class TestPhysicsInformedNN:
    """Tests for the PhysicsInformedNN class."""

    def test_default_initialization(self):
        model = PhysicsInformedNN()
        expected_nu = 0.01 / float(torch.pi)
        assert float(model.nu) == pytest.approx(expected_nu, rel=1e-5)

    def test_custom_nu(self):
        model = PhysicsInformedNN(nu=0.1)
        assert model.nu == 0.1

    def test_custom_architecture(self):
        model = PhysicsInformedNN(hidden_size=32, num_layers=6)
        # Count linear layers: 1 input + (num_layers - 1) hidden + 1 output = num_layers + 1
        linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 7  # 6 + 1 output
        assert linear_layers[0].in_features == 2
        assert linear_layers[0].out_features == 32
        assert linear_layers[-1].out_features == 1

    def test_default_layer_count(self):
        model = PhysicsInformedNN()
        linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 5  # 4 hidden + 1 output

    def test_forward_output_shape(self):
        model = PhysicsInformedNN()
        x = torch.rand(50, 1)
        t = torch.rand(50, 1)
        output = model(x, t)
        assert output.shape == (50, 1)

    def test_forward_single_point(self):
        model = PhysicsInformedNN()
        x = torch.tensor([[0.5]])
        t = torch.tensor([[0.3]])
        output = model(x, t)
        assert output.shape == (1, 1)
        assert torch.isfinite(output).all()

    def test_forward_batch(self):
        model = PhysicsInformedNN()
        batch_size = 200
        x = torch.rand(batch_size, 1)
        t = torch.rand(batch_size, 1)
        output = model(x, t)
        assert output.shape == (batch_size, 1)
        assert torch.isfinite(output).all()

    def test_physics_loss_returns_scalar(self):
        model = PhysicsInformedNN()
        x = torch.rand(100, 1)
        t = torch.rand(100, 1)
        loss = model.physics_loss(x, t)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_physics_loss_is_differentiable(self):
        model = PhysicsInformedNN()
        x = torch.rand(50, 1)
        t = torch.rand(50, 1)
        loss = model.physics_loss(x, t)
        loss.backward()
        # Check that at least some gradients are non-None
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads

    def test_eval_mode_produces_deterministic_output(self):
        model = PhysicsInformedNN()
        model.eval()
        x = torch.tensor([[0.0]])
        t = torch.tensor([[0.5]])
        with torch.no_grad():
            out1 = model(x, t).item()
            out2 = model(x, t).item()
        assert out1 == out2

    def test_model_state_dict_round_trip(self, tmp_path):
        model1 = PhysicsInformedNN(hidden_size=16, num_layers=3)
        path = tmp_path / "test_model.pth"
        torch.save(model1.state_dict(), str(path))

        model2 = PhysicsInformedNN(hidden_size=16, num_layers=3)
        model2.load_state_dict(torch.load(str(path), weights_only=True))

        x = torch.rand(10, 1)
        t = torch.rand(10, 1)
        with torch.no_grad():
            out1 = model1(x, t)
            out2 = model2(x, t)
        assert torch.allclose(out1, out2)

    def test_uses_tanh_activation(self):
        model = PhysicsInformedNN()
        tanh_layers = [m for m in model.net if isinstance(m, torch.nn.Tanh)]
        assert len(tanh_layers) >= 1

    def test_output_bounded_for_reasonable_inputs(self):
        """Untrained model output should still be finite for in-domain inputs."""
        torch.manual_seed(0)
        model = PhysicsInformedNN()
        x = torch.linspace(-1, 1, 100).view(-1, 1)
        t = torch.full((100, 1), 0.5)
        with torch.no_grad():
            out = model(x, t)
        assert torch.isfinite(out).all()
