"""Tests: POST /backtest/optimize contract — success and insufficient_data paths."""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app
from src.backtest.robust import OptimizationRun, CostAssumptions, WalkForwardConfig

client = TestClient(app)

VALID_PAYLOAD = {
    "ticker": "AAPL",
    "investment_horizon": "중기",
    "risk_profile": "중립형",
    "start": "2022-01-01",
    "end": "2024-12-31",
    "initial_balance": 10000,
    "n_trials": 2,
    "walk_forward": {"train_window_days": 180, "test_window_days": 60, "step_days": 60},
    "costs": {"fee_pct_one_way": 0.05, "slippage_pct_one_way": 0.05},
}


def _mock_df():
    from tests.fixtures.optimization_data import synthetic_prices
    return synthetic_prices(n=800)


def test_optimize_endpoint_exists():
    resp = client.post("/backtest/optimize", json={})
    assert resp.status_code in (400, 422)  # validation error, not 404


def test_optimize_returns_run_id_and_status():
    with patch("main.load_price_history", return_value=_mock_df()):
        resp = client.post("/backtest/optimize", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    assert "status" in body


def test_optimize_insufficient_data_returns_no_candidate():
    short_payload = {**VALID_PAYLOAD, "start": "2024-01-01", "end": "2024-03-31"}
    with patch("main.load_price_history", return_value=_mock_df().iloc[:30]):
        resp = client.post("/backtest/optimize", json=short_payload)
    body = resp.json()
    assert body["status"] == "insufficient_data"
    assert body.get("optimization", {}).get("robust_candidate") is None


def test_optimize_rejects_invalid_dates():
    bad = {**VALID_PAYLOAD, "start": "2024-12-31", "end": "2022-01-01"}
    resp = client.post("/backtest/optimize", json=bad)
    assert resp.status_code == 422


def test_optimize_rejects_n_trials_above_50():
    bad = {**VALID_PAYLOAD, "n_trials": 51}
    resp = client.post("/backtest/optimize", json=bad)
    assert resp.status_code == 422


def test_optimize_response_contains_warnings():
    with patch("main.load_price_history", return_value=_mock_df()):
        resp = client.post("/backtest/optimize", json=VALID_PAYLOAD)
    body = resp.json()
    opt = body.get("optimization", {})
    assert "warnings" in opt
