"""Cliente HTTP de la API P3 (única fuente de datos del agente).

Los endpoints están cableados a mano (``/openapi.json`` no existe en producción). Los errores de P3
(``{"error": {code, message, details, request_id}}``) y los fallos de red se convierten en ``ToolError``
con un mensaje apto para el usuario: nunca se propaga un stack trace.

Límites: el cuerpo se lee en streaming y se aborta al superar ``MAX_BODY_BYTES`` (también si
``Content-Length`` ya lo declara); además de ``timeout`` (por operación de red) hay un plazo total
(``total_timeout``) para toda la consulta, reintentos incluidos, que se comprueba tras cada trozo recibido.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable

import httpx

from agente.config import Settings, validate_url
from agente.formatting import clean

MAX_BODY_BYTES = 2_000_000
RETRY_STATUS = {502, 503, 504}


class ToolError(Exception):
    """Error controlado: ``message`` se puede mostrar tal cual al usuario."""

    def __init__(self, code: str, message: str, status: int | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.request_id = code, message, status, request_id


def _too_large() -> ToolError:
    return ToolError("response_too_large", "La API devolvió una respuesta demasiado grande.")


def _deadline_error() -> ToolError:
    return ToolError("timeout", "La API de mercado excedió el plazo total de respuesta.")


class P3Client:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        validate_url("P3_BASE_URL", settings.p3_base_url)
        # identity: sin compresión no hay bomba de descompresión (el límite de 2 MB se mide sobre bytes reales)
        headers = {"Accept": "application/json", "Accept-Encoding": "identity", "User-Agent": "agente-mercado/0.1"}
        if settings.p3_api_key:
            headers["X-API-Key"] = settings.p3_api_key
        # follow_redirects=False (por defecto): la clave nunca viaja a otro host
        self._http = httpx.Client(
            base_url=settings.p3_base_url, headers=headers, timeout=settings.timeout, transport=transport
        )
        self._retries = max(0, settings.retries)
        self._total = settings.total_timeout
        self._sleep = sleep
        self._clock = clock

    def close(self) -> None:
        self._http.close()

    def get(self, path: str, params: dict | None = None) -> dict | list:
        """GET con reintentos acotados (solo red/timeout y 502-504). ``path`` ya viene validado por ``tools``."""
        deadline = self._clock() + self._total
        last: ToolError | None = None
        for attempt in range(self._retries + 1):
            if attempt:
                self._sleep(0.3 * 2 ** (attempt - 1))
            if self._clock() > deadline:
                raise _deadline_error()
            try:
                status, headers, content = self._fetch(path, params, deadline)
            except httpx.TimeoutException:
                last = ToolError("timeout", "La API de mercado tardó demasiado en responder.")
                continue
            except httpx.TransportError:
                last = ToolError("api_unavailable", "No pude conectar con la API de mercado (¿está caída?).")
                continue
            if status in RETRY_STATUS and attempt < self._retries:
                last = self._error_from(status, headers, content)
                continue
            return self._parse(status, headers, content)
        raise last or ToolError("api_unavailable", "No pude conectar con la API de mercado (¿está caída?).")

    def _fetch(self, path: str, params: dict | None, deadline: float) -> tuple[int, httpx.Headers, bytes]:
        """Lee la respuesta por trozos: aborta al pasar MAX_BODY_BYTES o el plazo total, sin leer el resto."""
        with self._http.stream("GET", path, params=params) as resp:
            # Si el servidor comprime pese a Accept-Encoding: identity, se rechaza ANTES de leer/descomprimir nada
            if resp.headers.get("content-encoding", "identity").strip().lower() not in ("", "identity"):
                raise ToolError("unexpected_response", "Respuesta inesperada de la API (codificación no soportada).")
            declared = resp.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
                raise _too_large()
            buf = bytearray()
            for chunk in resp.iter_bytes():
                buf += chunk
                if len(buf) > MAX_BODY_BYTES:
                    raise _too_large()
                if self._clock() > deadline:
                    raise _deadline_error()
            return resp.status_code, resp.headers, bytes(buf)

    @staticmethod
    def _error_from(status: int, headers: httpx.Headers, content: bytes) -> ToolError:
        code, message, rid = "http_error", f"La API respondió con HTTP {status}.", None
        try:
            err = json.loads(content)["error"]
            code = clean(err.get("code") or code, 40)
            message = clean(err.get("message") or message, 300)
            rid = err.get("request_id")
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
        if status == 429:
            retry = headers.get("Retry-After")
            message = "La API limitó las peticiones (demasiadas en poco tiempo)." + (
                f" Reintenta en {retry[:6]} s." if retry and retry.isascii() and retry.isdigit() else ""
            )
        elif status >= 500 and code == "http_error":
            message = "La API de mercado tuvo un problema interno."
        return ToolError(code, message, status, clean(rid, 40) if isinstance(rid, str) else None)

    def _parse(self, status: int, headers: httpx.Headers, content: bytes) -> dict | list:
        if status >= 400:
            raise self._error_from(status, headers, content)
        if not content.strip():
            raise ToolError("unexpected_response", "Respuesta inesperada de la API (cuerpo vacío).")
        try:
            data = json.loads(content)
        except ValueError:
            raise ToolError("bad_response", "La API devolvió una respuesta que no es JSON.") from None
        if not isinstance(data, dict | list):
            raise ToolError("unexpected_response", "Respuesta inesperada de la API (formato no válido).")
        return data
