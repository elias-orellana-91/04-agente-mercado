"""P3 simulada con httpx.MockTransport: sin red, con las formas reales de respuesta de 03-api-mercado."""
from __future__ import annotations

import json

import httpx
import pytest

from agente.agent import Agent
from agente.client import P3Client
from agente.config import Settings
from agente.tools import Tools

TICKERS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Tecnología"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Tecnología"},
    {"symbol": "GS", "name": "The Goldman Sachs Group, Inc.", "sector": "Bancos"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "sector": "Índice"},
]


def err(status: int, code: str, message: str, headers: dict | None = None) -> httpx.Response:
    body = {"error": {"code": code, "message": message, "details": None, "request_id": "rid-123"}}
    return httpx.Response(status, json=body, headers=headers)


def item(rank, ticker, value):
    return {"rank": rank, "ticker": ticker, "name": f"{ticker} Inc.", "sector": "Tecnología", "value": value}


def make_handler(calls: list[httpx.Request], overrides: dict | None = None):
    overrides = overrides or {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path in overrides:
            r = overrides[path]
            return r(request) if callable(r) else r
        q = dict(request.url.params)
        if path == "/tickers":
            return httpx.Response(200, json=TICKERS)
        if path == "/ranking":
            vals = {"return": [0.071, 0.03, -0.01], "volatility": [0.55, 0.4, 0.2], "sharpe": [1.8, 1.1, 0.2]}[q["metric"]]
            if q["order"] == "asc":
                vals = vals[::-1]
            items = [item(i + 1, s, v) for i, (s, v) in enumerate(zip(["GS", "NVDA", "AAPL"], vals, strict=True))]
            return httpx.Response(200, json={"period": q["period"], "metric": q["metric"], "order": q["order"],
                                             "start": "2025-11-28", "end": "2025-12-30", "best": items[0], "items": items})
        if path == "/sectors/summary":
            return httpx.Response(200, json={"start": "2015-01-02", "end": "2025-12-30", "sectors": [
                {"sector": "Bancos", "tickers": ["GS"], "total_return": 2.5, "volatility": 0.28, "max_drawdown": -0.4,
                 "sharpe": 0.9, "best_ticker": "GS", "worst_ticker": "GS"},
                {"sector": "Tecnología", "tickers": ["AAPL", "NVDA"], "total_return": 9.1, "volatility": 0.33,
                 "max_drawdown": -0.35, "sharpe": 1.1, "best_ticker": "NVDA", "worst_ticker": "AAPL"}]})
        if path == "/model/info":
            return httpx.Response(200, json={"model_name": "logreg", "model_version": "0.1.0", "trained_at": "2026-10-01",
                                             "trained_on": "2015-2022", "sha256": "ab", "decision_threshold": 0.5,
                                             "features": ["a", "b"], "test_metrics": {"auc": 0.646, "accuracy": 0.6},
                                             "baselines_test_metrics": {"har_rv": {"auc": 0.627}}, "required_inputs": {}})
        if path.startswith("/metrics/"):
            return httpx.Response(200, json={"ticker": path.split("/")[-1], "start": "2015-01-02", "end": "2025-12-30",
                                             "n_days": 2700, "total_return": 9.1, "volatility": 0.318,
                                             "max_drawdown": -0.3057, "sharpe": 0.41})
        if path.startswith("/prices/"):
            return httpx.Response(200, json={"ticker": path.split("/")[-1], "start": None, "end": None, "total": 2,
                                             "limit": 5, "offset": 0, "items": [
                                                 {"date": "2025-12-30", "close": 273.5, "volume": 1},
                                                 {"date": "2025-12-29", "close": 272.0, "volume": 1}]})
        if path.startswith("/predictions/"):
            return httpx.Response(200, json={"ticker": path.split("/")[-1], "as_of_date": "2025-12-30", "horizon_days": 5,
                                             "probability_high_vol": 0.62, "predicted_class": 1, "label": "alta",
                                             "decision_threshold": 0.5, "model_version": "0.1.0", "in_sample": False,
                                             "created_at": "2026-01-02T10:00:00Z"})
        return err(404, "not_found", "Not Found")

    return handler


@pytest.fixture
def calls():
    return []


@pytest.fixture
def build(calls):
    """build(overrides=None, **settings) -> (agent, tools, calls)."""

    def _build(overrides=None, backend=None, **kw):
        settings = Settings(**{"retries": 2, **kw})
        client = P3Client(settings, transport=httpx.MockTransport(make_handler(calls, overrides)), sleep=lambda s: None)
        tools = Tools(client)
        return Agent(tools, backend), tools, calls

    return _build


@pytest.fixture
def agent(build):
    return build()[0]


def body(request: httpx.Request) -> dict:
    return json.loads(request.content)
