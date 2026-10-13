import os

import httpx
import pytest

from agente import cli
from agente.agent import Agent
from agente.client import P3Client
from agente.config import Settings
from agente.tools import Tools

from .conftest import make_handler


@pytest.fixture
def fake_p3(monkeypatch):
    def from_settings(cls, settings, backend=None):
        client = P3Client(settings, transport=httpx.MockTransport(make_handler([])), sleep=lambda s: None)
        return Agent(Tools(client))

    monkeypatch.setattr(Agent, "from_settings", classmethod(from_settings))


def test_cli_single_question(fake_p3, capsys):
    assert cli.main(["mejor", "acción", "del", "mes"]) == 0
    assert "GS" in capsys.readouterr().out


def test_cli_interactive(fake_p3, capsys, monkeypatch):
    lines = iter(["resumen por sector", "", "salir"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    assert cli.main([]) == 0
    assert "Tecnología" in capsys.readouterr().out


def test_cli_bad_backend_config(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["--backend", "claude", "hola"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_settings_from_env():
    s = Settings.from_env({"P3_BASE_URL": "http://x:1/", "P3_API_KEY": "CLAVE-XYZ", "AGENTE_BACKEND": "Ollama"})
    assert s.p3_base_url == "http://x:1" and s.backend == "ollama" and "CLAVE-XYZ" not in repr(s)


@pytest.mark.e2e
@pytest.mark.skipif(not os.environ.get("P3_E2E_URL"), reason="define P3_E2E_URL para probar contra una P3 real")
def test_e2e_against_real_p3():
    settings = Settings(p3_base_url=os.environ["P3_E2E_URL"].rstrip("/"), p3_api_key=os.environ.get("P3_API_KEY"))
    agent = Agent.from_settings(settings)
    assert "símbolos" in agent.ask("¿Qué acciones hay disponibles?")
    out = agent.ask("¿Cuál fue la mejor acción del mes?")
    assert "Ranking" in out, out
    out = agent.ask("¿Qué volatilidad se espera para NVDA?")
    assert "probabilidad" in out, out
    assert "Resumen por sector" in agent.ask("resumen por sector")
    assert "Modelo:" in agent.ask("¿cómo es el modelo?")
