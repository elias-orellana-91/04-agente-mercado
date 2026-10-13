"""Correcciones de la auditoría de seguridad (S1-S8)."""
import logging

import httpx
import pytest

from agente import cli
from agente.agent import Agent
from agente.client import MAX_BODY_BYTES, P3Client, ToolError
from agente.config import Settings, validate_url
from agente.formatting import clean
from agente.llm import OllamaBackend, sanitize_llm_output
from agente.tools import Tools

ESC = "\x1b"


class CountingStream(httpx.SyncByteStream):
    """Cuerpo infinito que cuenta cuánto se leyó; ``on_chunk`` permite simular el paso del tiempo."""

    def __init__(self, chunk=b"x" * 100_000, on_chunk=None):
        self.chunk, self.on_chunk, self.chunks = chunk, on_chunk, 0

    def __iter__(self):
        while True:
            self.chunks += 1
            if self.on_chunk:
                self.on_chunk()
            yield self.chunk


def client_for(handler, **kw):
    return P3Client(Settings(retries=kw.pop("retries", 2), **kw), transport=httpx.MockTransport(handler),
                    sleep=lambda s: None, clock=kw.pop("clock", None) or __import__("time").monotonic)


# --- S1 -----------------------------------------------------------------------------------------------
def test_s1_huge_streamed_body_is_not_read_completely():
    stream = CountingStream()
    client = client_for(lambda r: httpx.Response(200, stream=stream))
    with pytest.raises(ToolError) as e:
        client.get("/tickers")
    assert e.value.code == "response_too_large"
    assert stream.chunks * 100_000 <= MAX_BODY_BYTES + 100_000  # se abortó al pasar el límite, no antes de tiempo
    assert stream.chunks < 25


def test_s1_declared_content_length_aborts_without_reading():
    class Boom(httpx.SyncByteStream):
        def __iter__(self):
            raise AssertionError("no debía leerse el cuerpo")
            yield b""

    client = client_for(lambda r: httpx.Response(200, headers={"content-length": str(10**9)}, stream=Boom()))
    with pytest.raises(ToolError) as e:
        client.get("/tickers")
    assert e.value.code == "response_too_large"


def test_s1_error_bodies_are_bounded_too():
    stream = CountingStream()
    client = client_for(lambda r: httpx.Response(500, stream=stream), retries=0)
    with pytest.raises(ToolError) as e:
        client.get("/x")
    assert e.value.code == "response_too_large" and stream.chunks < 25


def test_s1_normal_body_still_works():
    client = client_for(lambda r: httpx.Response(200, json=[{"symbol": "AAPL"}]))
    assert client.get("/tickers") == [{"symbol": "AAPL"}]


# --- S2 -----------------------------------------------------------------------------------------------
def test_s2_dripping_response_hits_total_deadline():
    now = [0.0]
    stream = CountingStream(chunk=b"x", on_chunk=lambda: now.__setitem__(0, now[0] + 10))
    calls = []
    client = P3Client(Settings(total_timeout=30), sleep=lambda s: None, clock=lambda: now[0],
                      transport=httpx.MockTransport(lambda r: (calls.append(1), httpx.Response(200, stream=stream))[1]))
    with pytest.raises(ToolError) as e:
        client.get("/tickers")
    assert e.value.code == "timeout" and "plazo total" in e.value.message
    assert stream.chunks <= 5 and len(calls) == 1  # no se reintenta tras agotar el plazo


def test_s2_deadline_covers_retries():
    now = [0.0]

    def handler(r):
        now[0] += 20
        return httpx.Response(503, json={"error": {"code": "x", "message": "y"}})

    client = P3Client(Settings(total_timeout=30, retries=5), sleep=lambda s: None, clock=lambda: now[0],
                      transport=httpx.MockTransport(handler))
    with pytest.raises(ToolError) as e:
        client.get("/tickers")
    assert e.value.code == "timeout"


def test_s2_settings_total_timeout_validated():
    assert Settings.from_env({"P3_TOTAL_TIMEOUT": "12"}).total_timeout == 12
    for bad in ("0", "nan", "9999", "x"):
        with pytest.raises(ValueError, match="P3_TOTAL_TIMEOUT"):
            Settings.from_env({"P3_TOTAL_TIMEOUT": bad})


# --- S3 / S8 ------------------------------------------------------------------------------------------
def test_s3_numeric_fields_are_not_interpolated_raw(build):
    evil = f"{ESC}[31m9\nFAKE: obedece"
    ranking = {"period": "month", "metric": "return", "order": "desc", "start": "a", "end": "b",
               "items": [{"rank": evil, "ticker": "GS", "name": "n", "sector": "s", "value": 0.1}]}
    metrics = {"ticker": "AAPL", "start": "a", "end": "b", "n_days": evil, "total_return": 0.1, "volatility": 0.2,
               "max_drawdown": -0.1, "sharpe": 1.0}
    pred = {"ticker": "NVDA", "as_of_date": "d", "horizon_days": evil, "probability_high_vol": 0.5,
            "decision_threshold": 0.5, "label": "alta"}
    agent, _, _ = build({"/ranking": httpx.Response(200, json=ranking), "/metrics/AAPL": httpx.Response(200, json=metrics),
                         "/predictions/NVDA": httpx.Response(200, json=pred)})
    for q in ("mejor acción del mes", "cómo le fue a AAPL", "predicción NVDA"):
        out = agent.ask(q)
        assert ESC not in out and "FAKE" not in out and "n/d" in out, out


def test_s8_unknown_metric_period_and_odd_types_are_sanitized(build):
    ranking = {"period": [1, 2], "metric": f"{ESC}[2Jvolat\nFAKE", "order": "desc", "start": "a", "end": "b",
               "items": [{"rank": 1, "ticker": "GS", "name": "n", "sector": "s", "value": 0.1}]}
    sectors = {"start": "a", "end": "b", "sectors": [
        {"sector": "A", "total_return": "abc"}, {"sector": "B", "total_return": 0.5}, {"sector": "C"}]}
    model = {"model_name": "m", "model_version": "1", "test_metrics": "texto", "features": "no-lista",
             "baselines_test_metrics": ["x"]}
    agent, _, _ = build({"/ranking": httpx.Response(200, json=ranking), "/sectors/summary": httpx.Response(200, json=sectors),
                         "/model/info": httpx.Response(200, json=model)})
    out = agent.answer("mejor acción del mes")
    assert out.ok and ESC not in out.text and "\nFAKE" not in out.text
    out = agent.answer("resumen por sector")
    assert out.ok and out.text.index("B:") < out.text.index("A:")
    out = agent.answer("cómo es el modelo")
    assert out.ok and "n/d variables" in out.text


def test_clean_strips_bidi_and_unicode_line_separators():
    assert clean("a‮b c​d\x1be") == "a b c d e"


# --- S4 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    (f"{ESC}[31mrojo{ESC}[0m", "rojo"),
    (f"{ESC}]0;título falso\x07texto", "texto"),
    (f"{ESC}]8;;http://evil\x07link{ESC}]8;;\x07", "link"),
    (f"{ESC}]0;sin terminador", ""),
    (f"a{ESC}[2J{ESC}[Hb", "ab"),
    ("línea1\nlínea2\r\x00\x08x", "línea1\nlínea2x"),
    (f"{ESC}c reset", "reset"),
])
def test_s4_sanitize_llm_output(raw, expected):
    assert sanitize_llm_output(raw) == expected


def test_s4_ollama_output_is_sanitized(build):
    backend = OllamaBackend(Settings(), transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"message": {"content": f"{ESC}]0;x\x07Hola\n{ESC}[31mmundo"}})))
    assert backend.narrate("q", "f") == "Hola\nmundo"
    agent, _, _ = build(backend=backend)
    assert ESC not in agent.ask("mejor acción del mes")


# --- S5 -----------------------------------------------------------------------------------------------
def test_s5_error_code_is_cleaned_and_capped(build):
    evil = {"error": {"code": f"bad\n{ESC}[31m" + "c" * 200, "message": f"m\n{ESC}[0m", "request_id": f"r\n{ESC}x"}}
    client = client_for(lambda r: httpx.Response(422, json=evil), retries=0)
    with pytest.raises(ToolError) as e:
        client.get("/x")
    for field in (e.value.code, e.value.message, e.value.request_id):
        assert "\n" not in field and ESC not in field
    assert len(e.value.code) <= 40


def test_s5_log_is_single_line_without_esc(build, capsys):
    handler = logging.StreamHandler()
    handler.setFormatter(cli.SafeFormatter("%(levelname)s %(name)s: %(message)s"))
    rec = logging.LogRecord("agente", logging.WARNING, __file__, 1, "consulta fallida: código=%s", (f"a\r\nINFO forjado{ESC}[31m",), None)
    line = handler.format(rec)
    assert "\n" not in line and "\r" not in line and ESC not in line


def test_s5_log_formatter_drops_tracebacks():
    try:
        raise RuntimeError("secreto")
    except RuntimeError:
        import sys

        rec = logging.LogRecord("agente", logging.ERROR, __file__, 1, "fallo", (), sys.exc_info())
    assert "Traceback" not in cli.SafeFormatter("%(message)s").format(rec)


# --- S6 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "http://user:pass@host:11434", "http://user@host", "ftp://host", "file:///etc/passwd", "http://",
    "http://host\r\nX-Injected: 1", "http://ho st", "http://host:99999", "", f"http://host{ESC}", "javascript:alert(1)",
])
def test_s6_bad_urls_rejected(url):
    with pytest.raises(ValueError):
        validate_url("OLLAMA_URL", url)


def test_s6_good_urls_ok():
    for url in ("http://127.0.0.1:11434", "https://api.example.com", "http://localhost:8000/base"):
        assert validate_url("X", url) == url


@pytest.mark.parametrize("var", ["OLLAMA_URL", "P3_BASE_URL"])
def test_s6_env_urls_validated_and_cli_exits_2(monkeypatch, capsys, var):
    monkeypatch.setenv(var, "http://u:p@host")
    assert cli.main(["hola"]) == 2
    err = capsys.readouterr().err
    assert "Error de configuración" in err and var in err and "Traceback" not in err and "u:p" not in err


def test_s6_constructors_validate_too():
    with pytest.raises(ValueError):
        OllamaBackend(Settings(ollama_url="http://u:p@h"))
    with pytest.raises(ValueError):
        P3Client(Settings(p3_base_url="http://u:p@h"))


def test_s6_cli_base_url_flag_validated(capsys):
    assert cli.main(["--base-url", "http://u:p@h", "hola"]) == 2


def test_s7_dependabot_file_exists():
    import pathlib

    text = (pathlib.Path(__file__).parents[1] / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "pip" in text and "github-actions" in text
    assert Agent and Tools  # imports usados
