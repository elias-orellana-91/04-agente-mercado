"""Configuración del agente, leída de variables de entorno (nada de secretos en el código)."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

MAX_QUESTION_LEN = 500
MAX_TIMEOUT = 120.0
MAX_RETRIES = 5


def _number(name: str, raw, kind, lo, hi):
    """Convierte y valida un número de entorno (rechaza texto, nan, inf y valores fuera de rango)."""
    try:
        value = kind(str(raw).strip())
    except (ValueError, TypeError):
        raise ValueError(f"{name} debe ser un número (recibí '{str(raw)[:20]}').") from None
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} debe estar entre {lo} y {hi}.")
    return value


@dataclass(frozen=True)
class Settings:
    p3_base_url: str = "http://127.0.0.1:8000"
    p3_api_key: str | None = field(default=None, repr=False)  # nunca se imprime ni se loguea
    timeout: float = 10.0
    retries: int = 2  # reintentos extra (3 intentos en total) solo para fallos transitorios
    backend: str = "rules"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    claude_model: str = "claude-sonnet-5-5"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        """Lee el entorno; valores inválidos → ``ValueError`` con un mensaje apto para el usuario."""
        e = os.environ if env is None else env
        return cls(
            p3_base_url=e.get("P3_BASE_URL", cls.p3_base_url).strip().rstrip("/"),
            p3_api_key=e.get("P3_API_KEY") or None,
            timeout=_number("P3_TIMEOUT", e.get("P3_TIMEOUT", cls.timeout), float, 0.1, MAX_TIMEOUT),
            retries=_number("P3_RETRIES", e.get("P3_RETRIES", cls.retries), int, 0, MAX_RETRIES),
            backend=e.get("AGENTE_BACKEND", cls.backend).strip().lower(),
            ollama_url=e.get("OLLAMA_URL", cls.ollama_url).strip().rstrip("/"),
            ollama_model=e.get("OLLAMA_MODEL", cls.ollama_model),
            claude_model=e.get("AGENTE_CLAUDE_MODEL", cls.claude_model),
        )
