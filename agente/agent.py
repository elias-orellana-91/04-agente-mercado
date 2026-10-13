"""Agente: valida la pregunta, enruta, ejecuta herramientas, redacta y añade el aviso."""
from __future__ import annotations

import logging

from agente import formatting as F
from agente.client import P3Client, ToolError
from agente.config import MAX_QUESTION_LEN, Settings
from agente.llm import LLMBackend, LLMError, RulesBackend
from agente.router import Plan, plan
from agente.tools import Tools

log = logging.getLogger("agente")


class Agent:
    def __init__(self, tools: Tools, backend: LLMBackend | None = None) -> None:
        self.tools = tools
        self.backend = backend or RulesBackend()

    @classmethod
    def from_settings(cls, settings: Settings, backend: LLMBackend | None = None) -> Agent:
        from agente.llm import make_backend

        return cls(Tools(P3Client(settings)), backend or make_backend(settings))

    def ask(self, question: str) -> str:
        question = (question or "").strip()
        if not question:
            return "Escribe una pregunta sobre el mercado."
        if len(question) > MAX_QUESTION_LEN:
            return f"La pregunta es demasiado larga (máximo {MAX_QUESTION_LEN} caracteres)."
        try:
            p = plan(question, self.tools)
            if p.intent in ("clarify", "help"):
                return p.message
            facts = self._run(p)
        except ToolError as err:
            return F.error_text(err)
        except Exception:  # noqa: BLE001 - el usuario nunca ve un stack trace
            log.exception("Fallo inesperado del agente")
            return "Ocurrió un error inesperado al procesar la pregunta."
        text = facts
        if not isinstance(self.backend, RulesBackend):
            try:
                text = self.backend.narrate(question, facts)
            except LLMError as err:
                log.warning("Backend %s falló: %s", self.backend.name, err)
                text = facts + f"\n(No pude usar el backend '{self.backend.name}'; muestro la respuesta base.)"
        return f"{text}\n\n{F.DISCLAIMER}"

    def _run(self, p: Plan) -> str:
        t = self.tools
        if p.intent == "ranking":
            return F.ranking_text(t.ranking(**p.params))
        if p.intent == "sectors":
            return F.sectors_text(t.sectors_summary())
        if p.intent == "model":
            return F.model_text(t.model_info())
        if p.intent == "tickers":
            return F.tickers_text(t.tickers())
        per_symbol = {
            "prediction": lambda s: F.prediction_text(t.predict_volatility(s)),
            "metrics": lambda s: F.metrics_text(t.metrics(s)),
            "prices": lambda s: F.prices_text(t.prices(s)),
        }[p.intent]
        return "\n\n".join(per_symbol(s) for s in p.symbols)
