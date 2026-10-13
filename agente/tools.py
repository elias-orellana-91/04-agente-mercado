"""Herramientas del agente: una función por endpoint de P3, con validación estricta de argumentos.

Los símbolos se validan contra la lista real de tickers (``/tickers``) antes de armar cualquier URL, de
modo que nada que escriba el usuario llega a una ruta sin pasar por esa lista cerrada.
"""
from __future__ import annotations

import re
from datetime import date

from agente.client import P3Client, ToolError

SYMBOL_RE = re.compile(r"[A-Za-z][A-Za-z0-9.\-]{0,15}")
PERIODS = ("month", "quarter", "ytd")
METRICS = ("return", "volatility", "sharpe")
ORDERS = ("desc", "asc")


def _iso(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except (ValueError, TypeError):
        raise ToolError("invalid_argument", f"'{name}' debe ser una fecha YYYY-MM-DD.") from None


def _expect(data, keys: tuple[str, ...], lists: tuple[str, ...] = ()) -> dict:
    """Comprueba que la respuesta sea un objeto con los campos esperados (y listas de objetos donde toca)."""
    ok = isinstance(data, dict) and all(data.get(k) is not None for k in keys)
    ok = ok and all(isinstance(data.get(k), list) and all(isinstance(i, dict) for i in data[k]) for k in lists)
    if not ok:
        raise ToolError("unexpected_response", "Respuesta inesperada de la API (faltan campos esperados).")
    return data


class Tools:
    def __init__(self, client: P3Client) -> None:
        self.client = client
        self._tickers: list[dict] | None = None

    # --- tickers y validación -------------------------------------------------------------------------
    def tickers(self) -> list[dict]:
        if self._tickers is None:
            data = self.client.get("/tickers")
            if not isinstance(data, list):
                raise ToolError("bad_response", "La lista de tickers tiene un formato inesperado.")
            self._tickers = [t for t in data if isinstance(t, dict) and isinstance(t.get("symbol"), str)]
        return self._tickers

    def known_symbols(self) -> set[str]:
        return {t["symbol"].upper() for t in self.tickers()}

    def validate_symbol(self, symbol: str) -> str:
        if not isinstance(symbol, str) or not SYMBOL_RE.fullmatch(symbol.strip()):
            raise ToolError("invalid_symbol", "El símbolo no tiene un formato válido.")
        sym = symbol.strip().upper()
        if sym not in self.known_symbols():
            raise ToolError("ticker_not_found", f"El ticker '{sym}' no está en el universo de la API.")
        return sym

    # --- datos de mercado -----------------------------------------------------------------------------
    def prices(self, symbol: str, start: str | None = None, end: str | None = None, limit: int = 5,
               order: str = "desc") -> dict:
        sym = self.validate_symbol(symbol)
        if order not in ORDERS:
            raise ToolError("invalid_argument", "order debe ser 'asc' o 'desc'.")
        params = {"start": _iso(start, "start"), "end": _iso(end, "end"), "limit": max(1, min(int(limit), 500)),
                  "order": order}
        data = self.client.get(f"/prices/{sym}", {k: v for k, v in params.items() if v is not None})
        return _expect(data, ("ticker",), ("items",))

    def metrics(self, symbol: str, start: str | None = None, end: str | None = None) -> dict:
        sym = self.validate_symbol(symbol)
        params = {"start": _iso(start, "start"), "end": _iso(end, "end")}
        data = self.client.get(f"/metrics/{sym}", {k: v for k, v in params.items() if v is not None})
        return _expect(data, ("ticker", "start", "end", "total_return", "volatility", "max_drawdown"))

    def ranking(self, period: str = "month", metric: str = "return", order: str = "desc", limit: int = 5) -> dict:
        if period not in PERIODS or metric not in METRICS or order not in ORDERS:
            raise ToolError("invalid_argument", "Parámetros de ranking inválidos.")
        params = {"period": period, "metric": metric, "order": order, "limit": max(1, min(int(limit), 100))}
        return _expect(self.client.get("/ranking", params), ("period", "metric", "order", "start", "end"), ("items",))

    def sectors_summary(self, start: str | None = None, end: str | None = None) -> dict:
        params = {"start": _iso(start, "start"), "end": _iso(end, "end")}
        data = self.client.get("/sectors/summary", {k: v for k, v in params.items() if v is not None})
        return _expect(data, ("start", "end"), ("sectors",))

    # --- modelo ---------------------------------------------------------------------------------------
    def predict_volatility(self, symbol: str, on_date: str | None = None) -> dict:
        """Probabilidad de volatilidad alta a 5 días hábiles. Incluye ``in_sample`` (ver P3)."""
        sym = self.validate_symbol(symbol)
        params = {"date": _iso(on_date, "date")}
        data = self.client.get(f"/predictions/{sym}", {k: v for k, v in params.items() if v is not None})
        return _expect(data, ("ticker", "as_of_date", "probability_high_vol", "decision_threshold", "label"))

    def model_info(self) -> dict:
        return _expect(self.client.get("/model/info"), ("model_name", "model_version", "test_metrics"))
