"""Agente: valida la pregunta, enruta, ejecuta herramientas, redacta y añade el aviso."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from agente import formatting as F
from agente.client import P3Client, ToolError
from agente.config import MAX_QUESTION_LEN, Settings
from agente.llm import LLMBackend, LLMError, RulesBackend
from agente.router import Plan, plan
from agente.tools import Tools

log = logging.getLogger("agente")
IN_SAMPLE_MARK = "muestra de entrenamiento"


@dataclass(frozen=True)
class Answer:
    text: str
    ok: bool  # False si la consulta no se pudo responder (error de API/red, pregunta inválida o no entendida)


class Agent:
    def __init__(self, tools: Tools, backend: LLMBackend | None = None) -> None:
        self.tools = tools
        self.backend = backend or RulesBackend()

    @classmethod
    def from_settings(cls, settings: Settings, backend: LLMBackend | None = None) -> Agent:
        from agente.llm import make_backend

        return cls(Tools(P3Client(settings)), backend or make_backend(settings))

    def ask(self, question: str) -> str:
        return self.answer(question).text

    def answer(self, question: str) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer("Escribe una pregunta sobre el mercado.", False)
        if len(question) > MAX_QUESTION_LEN:
            return Answer(f"La pregunta es demasiado larga (máximo {MAX_QUESTION_LEN} caracteres).", False)
        try:
            p = plan(question, self.tools)
            if p.intent in ("clarify", "help"):
                return Answer(p.message, False)
            facts = self._run(p)
        except ToolError as err:
            log.warning("consulta fallida: código=%s", err.code)
            return Answer(F.error_text(err), False)
        except Exception as exc:  # noqa: BLE001 - el usuario nunca ve un stack trace
            log.error("fallo inesperado del agente: %s", type(exc).__name__)
            return Answer("Ocurrió un error inesperado al procesar la pregunta.", False)
        text = facts
        if not isinstance(self.backend, RulesBackend):
            try:
                text = self.backend.narrate(question, facts)
            except LLMError as err:
                log.warning("backend %s falló: %s", self.backend.name, err)
                text = facts + f"\n(No pude usar el backend '{self.backend.name}'; muestro la respuesta base.)"
            else:
                text = self._keep_in_sample_warning(facts, text)
        return Answer(f"{text}\n\n{F.DISCLAIMER}", True)

    @staticmethod
    def _keep_in_sample_warning(facts: str, text: str) -> str:
        """Si los hechos traen la advertencia in_sample y el LLM no la conservó, se anexa tal cual."""
        if IN_SAMPLE_MARK in text:
            return text
        warnings = [ln for ln in facts.splitlines() if ln.startswith("Atención:") and IN_SAMPLE_MARK in ln]
        return text + "\n" + "\n".join(warnings) if warnings else text

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
