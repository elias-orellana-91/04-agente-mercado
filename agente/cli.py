"""CLI: ``python -m agente "pregunta"`` o modo interactivo (sin argumentos).

Códigos de salida: 0 = respondida; 1 = la consulta no se pudo responder (error de API/red, pregunta vacía,
demasiado larga, no entendida o símbolo desconocido); 2 = configuración o uso incorrecto.
El modo interactivo termina siempre con 0.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from agente.agent import Agent
from agente.config import Settings
from agente.formatting import clean
from agente.llm import LLMError


class SafeFormatter(logging.Formatter):
    """Una línea por registro: sin saltos de línea ni caracteres de control (ESC, CR...) en el mensaje."""

    def format(self, record: logging.LogRecord) -> str:
        record.exc_info = record.exc_text = None  # nunca tracebacks en la salida
        return clean(super().format(record), 500)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(SafeFormatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    ap = argparse.ArgumentParser(prog="agente", description="Agente analista de mercado (consume la API P3).")
    ap.add_argument("pregunta", nargs="*", help="Pregunta en español; sin ella se abre el modo interactivo.")
    ap.add_argument("--backend", choices=["rules", "ollama", "claude"], help="Por defecto AGENTE_BACKEND o 'rules'.")
    ap.add_argument("--base-url", help="URL de P3 (por defecto P3_BASE_URL o http://127.0.0.1:8000).")
    args = ap.parse_args(argv)  # uso incorrecto → argparse sale con código 2

    try:
        settings = Settings.from_env()
        overrides = {k: v for k, v in {"backend": args.backend,
                                       "p3_base_url": args.base_url and args.base_url.rstrip("/")}.items() if v}
        settings = replace(settings, **overrides)
        agent = Agent.from_settings(settings)
    except (LLMError, ValueError) as err:
        print(f"Error de configuración: {err}", file=sys.stderr)
        return 2

    if args.pregunta:
        result = agent.answer(" ".join(args.pregunta))
        print(result.text)
        return 0 if result.ok else 1
    print("Agente de mercado (escribe 'salir' para terminar).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in ("salir", "exit", "quit"):
            return 0
        if q:
            print(agent.ask(q), end="\n\n")
