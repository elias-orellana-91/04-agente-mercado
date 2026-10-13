"""Configuración del agente, leída de variables de entorno (nada de secretos en el código)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

MAX_QUESTION_LEN = 500


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
        e = os.environ if env is None else env
        return cls(
            p3_base_url=e.get("P3_BASE_URL", cls.p3_base_url).strip().rstrip("/"),
            p3_api_key=e.get("P3_API_KEY") or None,
            timeout=float(e.get("P3_TIMEOUT", cls.timeout)),
            retries=int(e.get("P3_RETRIES", cls.retries)),
            backend=e.get("AGENTE_BACKEND", cls.backend).strip().lower(),
            ollama_url=e.get("OLLAMA_URL", cls.ollama_url).strip().rstrip("/"),
            ollama_model=e.get("OLLAMA_MODEL", cls.ollama_model),
            claude_model=e.get("AGENTE_CLAUDE_MODEL", cls.claude_model),
        )
