import httpx
import pytest

from agente.client import P3Client, ToolError
from agente.config import Settings

from .conftest import err


def test_tickers_cached(build):
    _, tools, calls = build()
    assert len(tools.tickers()) == 4
    tools.tickers()
    assert sum(c.url.path == "/tickers" for c in calls) == 1


def test_metrics_url_and_params(build):
    _, tools, calls = build()
    m = tools.metrics("aapl", start="2025-01-01")
    assert m["ticker"] == "AAPL"
    req = calls[-1]
    assert req.url.path == "/metrics/AAPL" and req.url.params["start"] == "2025-01-01"


def test_prices(build):
    _, tools, calls = build()
    p = tools.prices("NVDA", limit=9999)
    assert p["items"] and calls[-1].url.params["limit"] == "500" and calls[-1].url.params["order"] == "desc"


def test_ranking_and_sectors_and_model(build):
    _, tools, calls = build()
    assert tools.ranking("quarter", "sharpe", "asc")["best"]["rank"] == 1
    assert calls[-1].url.params["period"] == "quarter"
    assert tools.sectors_summary()["sectors"]
    assert tools.model_info()["model_version"] == "0.1.0"


def test_predict_has_in_sample_field(build):
    _, tools, _ = build()
    assert tools.predict_volatility("NVDA")["in_sample"] is False


@pytest.mark.parametrize("bad", ["../etc/passwd", "AAPL/../x", "AAPL?x=1", "A A", "", "ZZZZ", "AAPL%2f", "1AAPL", "a" * 40])
def test_symbol_injection_and_unknown_never_reach_url(build, bad):
    _, tools, calls = build()
    with pytest.raises(ToolError) as e:
        tools.metrics(bad)
    assert e.value.code in ("invalid_symbol", "ticker_not_found")
    assert all(c.url.path == "/tickers" for c in calls)  # solo se consultó la lista, nunca /metrics/...


@pytest.mark.parametrize("call", [
    lambda t: t.ranking("decade"), lambda t: t.ranking(metric="x"), lambda t: t.ranking(order="up"),
    lambda t: t.metrics("AAPL", start="ayer"), lambda t: t.prices("AAPL", order="x"),
])
def test_invalid_arguments(build, call):
    _, tools, _ = build()
    with pytest.raises(ToolError) as e:
        call(tools)
    assert e.value.code == "invalid_argument"


def test_p3_error_envelope_is_parsed(build):
    _, tools, _ = build({"/metrics/AAPL": err(422, "insufficient_data", "El rango no tiene al menos 2 precios.")})
    with pytest.raises(ToolError) as e:
        tools.metrics("AAPL")
    assert (e.value.code, e.value.status, e.value.request_id) == ("insufficient_data", 422, "rid-123")


def test_rate_limit_message(build):
    _, tools, calls = build({"/ranking": err(429, "rate_limited", "x", {"Retry-After": "60"})})
    with pytest.raises(ToolError) as e:
        tools.ranking()
    assert "60" in e.value.message and e.value.code == "rate_limited"
    assert sum(c.url.path == "/ranking" for c in calls) == 1  # 429 no se reintenta


def test_retries_503_then_succeeds(build):
    state = {"n": 0}

    def flaky(_):
        state["n"] += 1
        return err(503, "database_unavailable", "x") if state["n"] < 3 else httpx.Response(200, json={"start": "a", "end": "b", "sectors": []})

    _, tools, _ = build({"/sectors/summary": flaky})
    assert tools.sectors_summary()["sectors"] == [] and state["n"] == 3


def test_retries_are_bounded(build):
    state = {"n": 0}

    def down(_):
        state["n"] += 1
        return err(503, "model_unavailable", "Modelo no disponible.")

    _, tools, _ = build({"/model/info": down})
    with pytest.raises(ToolError) as e:
        tools.model_info()
    assert state["n"] == 3 and e.value.code == "model_unavailable"


def test_404_not_retried(build):
    state = {"n": 0}

    def nf(_):
        state["n"] += 1
        return err(404, "ticker_not_found", "no existe")

    _, tools, _ = build({"/metrics/AAPL": nf})
    with pytest.raises(ToolError):
        tools.metrics("AAPL")
    assert state["n"] == 1


@pytest.mark.parametrize("exc,code", [(httpx.ConnectError("boom"), "api_unavailable"), (httpx.ReadTimeout("slow"), "timeout")])
def test_network_failures(exc, code):
    def handler(request):
        raise exc

    client = P3Client(Settings(retries=1), transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(ToolError) as e:
        client.get("/tickers")
    assert e.value.code == code


def test_non_json_and_non_envelope_errors():
    client = P3Client(Settings(retries=0), transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="<html>") if r.url.path == "/a" else httpx.Response(500, text="boom")))
    with pytest.raises(ToolError) as e:
        client.get("/a")
    assert e.value.code == "bad_response"
    with pytest.raises(ToolError) as e:
        client.get("/b")
    assert e.value.status == 500 and "boom" not in e.value.message


def test_api_key_sent_only_when_set_and_not_in_repr():
    seen = []
    t = httpx.MockTransport(lambda r: (seen.append(dict(r.headers)), httpx.Response(200, json=[]))[1])
    P3Client(Settings(p3_api_key="SECRET"), transport=t).get("/tickers")
    P3Client(Settings(), transport=t).get("/tickers")
    assert seen[0]["x-api-key"] == "SECRET" and "x-api-key" not in seen[1]
    assert "SECRET" not in repr(Settings(p3_api_key="SECRET"))


def test_bad_base_url():
    with pytest.raises(ValueError):
        P3Client(Settings(p3_base_url="file:///etc"))


@pytest.mark.parametrize("path,call", [
    ("/predictions/NVDA", lambda t: t.predict_volatility("NVDA")),
    ("/model/info", lambda t: t.model_info()),
    ("/metrics/AAPL", lambda t: t.metrics("AAPL")),
    ("/prices/AAPL", lambda t: t.prices("AAPL")),
    ("/sectors/summary", lambda t: t.sectors_summary()),
    ("/ranking", lambda t: t.ranking()),
])
@pytest.mark.parametrize("payload", [{}, [], None, {"unexpected": 1}, "texto"])
def test_unexpected_200_bodies_become_tool_error(build, path, call, payload):
    _, tools, _ = build({path: httpx.Response(200, json=payload)})
    with pytest.raises(ToolError) as e:
        call(tools)
    assert e.value.code == "unexpected_response"


def test_unexpected_body_message_through_agent(build):
    agent, _, _ = build({"/predictions/NVDA": httpx.Response(200, json={})})
    out = agent.answer("predicción NVDA")
    assert not out.ok and "Respuesta inesperada de la API" in out.text and "None" not in out.text


def test_ranking_items_must_be_list_of_objects(build):
    bad = {"period": "month", "metric": "return", "order": "desc", "start": "a", "end": "b", "items": [1, 2]}
    _, tools, _ = build({"/ranking": httpx.Response(200, json=bad)})
    with pytest.raises(ToolError):
        tools.ranking()