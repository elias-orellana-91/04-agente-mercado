"""Enrutado de preguntas en español a herramientas, por palabras clave (sin LLM).

``plan()`` solo decide QUÉ herramienta usar y con qué parámetros; no llama a la red salvo para conocer la
lista de tickers (a través de ``Tools``).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from agente.tools import Tools

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9.\-]{0,15}")
# Siglas en mayúsculas que NO son símbolos bursátiles
NOT_SYMBOLS = {"YTD", "USD", "AUC", "EEUU", "EE.UU", "ROC", "IA", "ML", "ETF", "API", "VIX", "HAR", "RV", "UU", "EE"}
MAX_SYMBOLS = 3


@dataclass
class Plan:
    intent: str  # ranking | prediction | model | sectors | tickers | prices | metrics | clarify | help
    params: dict = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    message: str = ""  # para clarify/help


def normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def extract_symbols(question: str, known: set[str]) -> tuple[list[str], list[str]]:
    """Devuelve (símbolos conocidos, siglas en mayúsculas desconocidas).

    Un token corto (<3 letras) solo cuenta si está escrito en mayúsculas, para no confundir 'a' o 'es'
    con un ticker.
    """
    found: list[str] = []
    unknown: list[str] = []
    for tok in WORD_RE.findall(question):
        up = tok.upper()
        if up in known:
            if (tok.isupper() or len(tok) >= 3) and up not in found:
                found.append(up)
        elif tok.isupper() and 2 <= len(tok) <= 5 and tok.isalpha() and up not in NOT_SYMBOLS and up not in unknown:
            unknown.append(up)
    return found, unknown


HELP = (
    "No entendí la pregunta. Puedo responder, por ejemplo:\n"
    "  - ¿Cuál fue la mejor acción del mes? (también trimestre / año, peor, más volátil, mejor Sharpe)\n"
    "  - ¿Qué volatilidad se espera para NVDA?\n"
    "  - Dame un resumen por sector\n"
    "  - ¿Cómo le fue a AAPL? / ¿Cuál es el precio de MSFT?\n"
    "  - ¿Qué acciones hay disponibles? / ¿Cómo es el modelo?"
)


def plan(question: str, tools: Tools) -> Plan:
    q = normalize(question)
    symbols, unknown = extract_symbols(question, tools.known_symbols())
    symbols = symbols[:MAX_SYMBOLS]

    predict = _has(q, r"\bespera", r"predic", r"pronostic", r"proyecci", r"futur", r"proximos? (dias|5)",
                   r"probabilidad", r"\bpredecir\b")
    if predict:
        if symbols:
            return Plan("prediction", symbols=symbols)
        if unknown:
            return Plan("clarify", message=f"No reconozco el símbolo {', '.join(unknown)} en la API.")
        return Plan("clarify", message="¿Para qué acción? Indica el símbolo, p. ej. «¿Qué volatilidad se espera para NVDA?».")

    if _has(q, r"\bmodelo\b", r"\bauc\b", r"entrenad", r"entrenamiento", r"\bfeatures\b"):
        return Plan("model")

    if _has(q, r"\bsector"):
        return Plan("sectors")

    if not symbols and _has(q, r"(que|cuales|lista|listar|disponibles|hay|soportad).*(acciones|tickers|simbolos|empresas)",
                            r"(acciones|tickers|simbolos|empresas).*(disponibles|hay|soportad|cubre)"):
        return Plan("tickers")

    metric = "sharpe" if "sharpe" in q else "volatility" if "volatil" in q else "return"
    superlative = _has(q, r"\b(mejor|mejores|peor|peores|top|ranking|ganador|perdedor|rindio|rentable)\b")
    extreme = metric != "return" and _has(q, r"\b(mas|menos|mayor|menor|maxim|minim)\b")
    if (superlative or extreme) and not (unknown and not symbols):
        period = "quarter" if _has(q, r"trimestre") else "ytd" if _has(q, r"\bytd\b", r"\bano\b", r"anual") else "month"
        asc = _has(q, r"\bpeor", r"\bmenos\b", r"\bmenor\b", r"minim", r"\bperdedor", r"\bmas bajo", r"mas baja")
        return Plan("ranking", params={"period": period, "metric": metric, "order": "asc" if asc else "desc"})

    if unknown and not symbols:
        return Plan("clarify", message=f"No reconozco el símbolo {', '.join(unknown)} en la API. "
                                       "Pregunta «¿Qué acciones hay disponibles?» para ver la lista.")

    if symbols and _has(q, r"precio", r"cierre", r"cotiza", r"cuanto vale", r"cuanto cuesta"):
        return Plan("prices", symbols=symbols)
    if symbols:
        return Plan("metrics", symbols=symbols)
    return Plan("help", message=HELP)
