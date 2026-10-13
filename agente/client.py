"""Cliente HTTP de la API P3 (única fuente de datos del agente).

Los endpoints están cableados a mano (``/openapi.json`` no existe en producción). Los errores de P3
(``{"error": {code, message, details, request_id}}``) y los fallos de red se convierten en ``ToolError``
con un mensaje apto para el usuario: nunca se propaga un stack trace.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlparse

import httpx

from agente.config import Settings

MAX_BODY_BYTES = 2_000_000
RETRY_STATUS = {502, 503, 504}


class ToolError(Exception):
    """Error controlado: ``message`` se puede mostrar tal cual al usuario."""

    def __init__(self, code: str, message: str, status: int | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.request_id = code, message, status, request_id


class P3Client:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        parsed = urlparse(settings.p3_base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("P3_BASE_URL debe ser una URL http(s) válida.")
        headers = {"Accept": "application/json", "User-Agent": "agente-mercado/0.1"}
        if settings.p3_api_key:
            headers["X-API-Key"] = settings.p3_api_key
        # follow_redirects=False (por defecto): la clave nunca viaja a otro host
        self._http = httpx.Client(
            base_url=settings.p3_base_url, headers=headers, timeout=settings.timeout, transport=transport
        )
        self._retries = max(0, settings.retries)
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    def get(self, path: str, params: dict | None = None) -> dict | list:
        """GET con reintentos acotados (solo red/timeout y 502-504). ``path`` ya viene validado por ``tools``."""
        last: ToolError | None = None
        for attempt in range(self._retries + 1):
            if attempt:
                self._sleep(0.3 * 2 ** (attempt - 1))
            try:
                resp = self._http.get(path, params=params)
            except httpx.TimeoutException:
                last = ToolError("timeout", "La API de mercado tardó demasiado en responder.")
                continue
            except httpx.TransportError:
                last = ToolError("api_unavailable", "No pude conectar con la API de mercado (¿está caída?).")
                continue
            if resp.status_code in RETRY_STATUS and attempt < self._retries:
                last = self._error_from(resp)
                continue
            return self._parse(resp)
        assert last is not None
        raise last

    @staticmethod
    def _error_from(resp: httpx.Response) -> ToolError:
        code, message, rid = "http_error", f"La API respondió con HTTP {resp.status_code}.", None
        try:
            err = resp.json()["error"]
            code = str(err.get("code") or code)
            message = str(err.get("message") or message)
            rid = err.get("request_id")
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After")
            message = "La API limitó las peticiones (demasiadas en poco tiempo)." + (
                f" Reintenta en {retry[:6]} s." if retry and retry.isdigit() else ""
            )
        elif resp.status_code >= 500 and code == "http_error":
            message = "La API de mercado tuvo un problema interno."
        return ToolError(code, message[:300], resp.status_code, rid if isinstance(rid, str) else None)

    def _parse(self, resp: httpx.Response) -> dict | list:
        if resp.status_code >= 400:
            raise self._error_from(resp)
        if len(resp.content) > MAX_BODY_BYTES:
            raise ToolError("response_too_large", "La API devolvió una respuesta demasiado grande.")
        try:
            data = resp.json()
        except ValueError:
            raise ToolError("bad_response", "La API devolvió una respuesta que no es JSON.") from None
        if not isinstance(data, dict | list):
            raise ToolError("bad_response", "La API devolvió una respuesta inesperada.")
        return data
