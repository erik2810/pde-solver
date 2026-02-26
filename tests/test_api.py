"""Tests for the FastAPI inference server."""

import pytest
import torch
from fastapi.testclient import TestClient

from equations import EQUATIONS
from model import PhysicsInformedNN


@pytest.fixture(autouse=True)
def _setup_models(tmp_path, monkeypatch):
    """Create temporary models for burgers and heat, point the server at them."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Save a small model for burgers (default architecture matches equation recommendation)
    burgers_eq = EQUATIONS["burgers"]
    burgers_model = PhysicsInformedNN(
        hidden_size=burgers_eq.recommended_hidden,
        num_layers=burgers_eq.recommended_layers,
    )
    torch.save(burgers_model.state_dict(), str(models_dir / "burgers_model.pth"))

    # Save a small model for heat
    heat_eq = EQUATIONS["heat"]
    heat_model = PhysicsInformedNN(
        hidden_size=heat_eq.recommended_hidden,
        num_layers=heat_eq.recommended_layers,
    )
    torch.save(heat_model.state_dict(), str(models_dir / "heat_model.pth"))

    monkeypatch.setenv("MODEL_DIR", str(models_dir))

    # Re-import to pick up the env var
    import importlib
    import main as main_mod

    importlib.reload(main_mod)
    # Manually trigger startup
    main_mod.startup_event()
    yield main_mod


@pytest.fixture
def client(_setup_models):
    return TestClient(_setup_models.app)


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_models_loaded(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"
        assert "burgers" in data["models_loaded"]
        assert "heat" in data["models_loaded"]
        assert data["models_available"] == 2
        assert data["uptime_seconds"] is not None

    def test_health_degraded_when_no_models(self, _setup_models):
        """When no models are loaded, status should be degraded."""
        main_mod = _setup_models
        # Directly clear all loaded models to simulate "no models available"
        main_mod.loaded_models.clear()

        c = TestClient(main_mod.app)
        data = c.get("/api/v1/health").json()
        assert data["status"] == "degraded"
        assert data["models_loaded"] == []
        assert data["models_available"] == 0


class TestEquationsEndpoint:
    """Tests for GET /api/v1/equations."""

    def test_returns_all_equations(self, client):
        resp = client.get("/api/v1/equations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == len(EQUATIONS)

    def test_equation_metadata_fields(self, client):
        data = client.get("/api/v1/equations").json()
        for eq in data:
            assert "key" in eq
            assert "name" in eq
            assert "description" in eq
            assert "latex_equation" in eq
            assert "trained" in eq

    def test_trained_status(self, client):
        data = client.get("/api/v1/equations").json()
        trained_keys = {eq["key"] for eq in data if eq["trained"]}
        assert "burgers" in trained_keys
        assert "heat" in trained_keys
        # wave was not saved, so it should not be trained
        untrained_keys = {eq["key"] for eq in data if not eq["trained"]}
        assert "wave" in untrained_keys


class TestPredictEndpoint:
    """Tests for POST /api/v1/predict/{equation_key}."""

    def test_valid_burgers_request(self, client):
        resp = client.post("/api/v1/predict/burgers", json={"t": 0.5})
        assert resp.status_code == 200
        data = resp.json()
        assert "x" in data
        assert "u" in data
        assert data["t"] == 0.5
        assert data["n_points"] == 100
        assert data["equation"] == "burgers"

    def test_valid_heat_request(self, client):
        resp = client.post("/api/v1/predict/heat", json={"t": 0.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["equation"] == "heat"

    def test_returns_correct_number_of_points(self, client):
        resp = client.post("/api/v1/predict/burgers", json={"t": 0.5, "n_points": 50})
        data = resp.json()
        assert len(data["x"]) == 50
        assert len(data["u"]) == 50
        assert data["n_points"] == 50

    def test_t_zero(self, client):
        resp = client.post("/api/v1/predict/burgers", json={"t": 0.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["t"] == 0.0

    def test_t_one(self, client):
        resp = client.post("/api/v1/predict/burgers", json={"t": 1.0})
        assert resp.status_code == 200

    def test_x_range_burgers(self, client):
        data = client.post("/api/v1/predict/burgers", json={"t": 0.5}).json()
        assert data["x"][0] == pytest.approx(-1.0)
        assert data["x"][-1] == pytest.approx(1.0)

    def test_all_values_finite(self, client):
        data = client.post("/api/v1/predict/burgers", json={"t": 0.5}).json()
        assert all(isinstance(v, float) for v in data["u"])

    def test_caching_returns_same_result(self, client):
        data1 = client.post("/api/v1/predict/burgers", json={"t": 0.42}).json()
        data2 = client.post("/api/v1/predict/burgers", json={"t": 0.42}).json()
        assert data1["u"] == data2["u"]

    def test_unknown_equation_404(self, client):
        resp = client.post("/api/v1/predict/nonexistent", json={"t": 0.5})
        assert resp.status_code == 404

    def test_untrained_equation_503(self, client):
        resp = client.post("/api/v1/predict/wave", json={"t": 0.5})
        assert resp.status_code == 503

    def test_invalid_t_below_range(self, client):
        resp = client.post("/api/v1/predict/burgers", json={"t": -0.1})
        assert resp.status_code == 422

    def test_invalid_t_above_range(self, client):
        resp = client.post("/api/v1/predict/burgers", json={"t": 1.5})
        assert resp.status_code == 422

    def test_invalid_n_points_too_low(self, client):
        resp = client.post(
            "/api/v1/predict/burgers", json={"t": 0.5, "n_points": 5}
        )
        assert resp.status_code == 422

    def test_invalid_n_points_too_high(self, client):
        resp = client.post(
            "/api/v1/predict/burgers", json={"t": 0.5, "n_points": 1000}
        )
        assert resp.status_code == 422

    def test_missing_t(self, client):
        resp = client.post("/api/v1/predict/burgers", json={})
        assert resp.status_code == 422

    def test_invalid_json(self, client):
        resp = client.post(
            "/api/v1/predict/burgers",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


class TestLegacyEndpoint:
    """Tests for the backward-compatible POST /predict_snapshot."""

    def test_legacy_endpoint_works(self, client):
        resp = client.post("/predict_snapshot", json={"t": 0.3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["x"]) == 100

    def test_legacy_v1_endpoint_works(self, client):
        resp = client.post("/api/v1/predict_snapshot", json={"t": 0.3})
        assert resp.status_code == 200

    def test_legacy_validation(self, client):
        resp = client.post("/predict_snapshot", json={"t": 2.0})
        assert resp.status_code == 422


class TestServeFrontend:
    """Tests for GET / serving the dashboard."""

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
