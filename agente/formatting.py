"""Plantillas de respuesta en español. El texto que viene de la API se trata como DATO: se sanea y acota."""
from __future__ import annotations

import re

DISCLAIMER = (
    "Aviso: información educativa basada en datos históricos y un modelo experimental; "
    "no es asesoría ni recomendación de inversión."
)
PERIOD_ES = {"month": "del mes", "quarter": "del trimestre", "ytd": "del año en curso", "custom": "del periodo"}
METRIC_ES = {"return": "retorno", "volatility": "volatilidad anualizada", "sharpe": "Sharpe"}
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def clean(value, n: int = 80) -> str:
    """Texto externo → una línea sin caracteres de control y con largo acotado."""
    return _CTRL.sub(" ", str(value)).strip()[:n]


def num(x, d: int = 2) -> str:
    if not isinstance(x, int | float) or isinstance(x, bool):
        return "n/d"
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d: int = 1, sign: bool = True) -> str:
    if not isinstance(x, int | float) or isinstance(x, bool):
        return "n/d"
    s = f"{x * 100:+.{d}f}" if sign else f"{x * 100:.{d}f}"
    return s.replace(".", ",") + " %"


def _value(metric: str, v) -> str:
    return num(v) if metric == "sharpe" else pct(v, sign=metric == "return")


def tickers_text(data: list[dict]) -> str:
    by_sector: dict[str, list[str]] = {}
    for t in data:
        by_sector.setdefault(clean(t.get("sector", "?"), 40), []).append(clean(t.get("symbol", "?"), 16))
    lines = [f"La API cubre {len(data)} símbolos:"]
    lines += [f"  - {s}: {', '.join(sorted(v))}" for s, v in sorted(by_sector.items())]
    return "\n".join(lines)


def ranking_text(r: dict) -> str:
    period, metric, order = r.get("period", "month"), r.get("metric", "return"), r.get("order", "desc")
    items = r.get("items") or []
    head = f"Ranking por {METRIC_ES.get(metric, metric)} {PERIOD_ES.get(period, '')} ({clean(r.get('start'), 10)} → {clean(r.get('end'), 10)}), orden {'descendente' if order == 'desc' else 'ascendente'}:"
    if not items:
        return "No hay datos suficientes para armar el ranking."
    lines = [head]
    for it in items:
        lines.append(
            f"  {it.get('rank')}. {clean(it.get('ticker'), 16)} ({clean(it.get('name'), 40)}, {clean(it.get('sector'), 30)}): "
            f"{_value(metric, it.get('value'))}"
        )
    first = items[0]
    tk, val = clean(first.get("ticker"), 16), _value(metric, first.get("value"))
    if metric == "return" and order == "desc":
        lead = f"La acción de mayor retorno {PERIOD_ES.get(period, '')} fue {tk} ({val})."
    elif metric == "return":
        lead = f"La acción de menor retorno {PERIOD_ES.get(period, '')} fue {tk} ({val})."
    else:
        lead = f"Primero del ranking ({'mayor' if order == 'desc' else 'menor'} {METRIC_ES[metric]}): {tk} ({val})."
    return lead + "\n" + "\n".join(lines)


def metrics_text(m: dict) -> str:
    return (
        f"{clean(m.get('ticker'), 16)} entre {clean(m.get('start'), 10)} y {clean(m.get('end'), 10)} ({m.get('n_days')} días): "
        f"retorno total {pct(m.get('total_return'))}, volatilidad anualizada {pct(m.get('volatility'), sign=False)}, "
        f"máxima caída {pct(m.get('max_drawdown'))}, Sharpe {num(m.get('sharpe'))}."
    )


def prices_text(p: dict) -> str:
    items = p.get("items") or []
    if not items:
        return f"No hay precios para {clean(p.get('ticker'), 16)} en ese rango."
    lines = [f"Últimos cierres ajustados de {clean(p.get('ticker'), 16)} (USD):"]
    lines += [f"  - {clean(i.get('date'), 10)}: {num(i.get('close'))}" for i in items]
    return "\n".join(lines)


def sectors_text(s: dict) -> str:
    sectors = s.get("sectors") or []
    if not sectors:
        return "No hay datos de sectores."
    lines = [f"Resumen por sector, cartera de igual ponderación ({clean(s.get('start'), 10)} → {clean(s.get('end'), 10)}):"]
    for x in sorted(sectors, key=lambda z: z.get("total_return") or -9, reverse=True):
        lines.append(
            f"  - {clean(x.get('sector'), 40)}: retorno {pct(x.get('total_return'))}, volatilidad {pct(x.get('volatility'), sign=False)}, "
            f"máx. caída {pct(x.get('max_drawdown'))}; mejor {clean(x.get('best_ticker'), 16)}, peor {clean(x.get('worst_ticker'), 16)}"
        )
    return "\n".join(lines)


def prediction_text(p: dict) -> str:
    prob = p.get("probability_high_vol")
    txt = (
        f"Para {clean(p.get('ticker'), 16)} (datos al {clean(p.get('as_of_date'), 10)}) el modelo estima una probabilidad de "
        f"{pct(prob, 0, sign=False)} de volatilidad ALTA en los próximos {p.get('horizon_days', 5)} días hábiles "
        f"(umbral {num(p.get('decision_threshold'))}; clasificación: {clean(p.get('label'), 10)})."
    )
    if p.get("in_sample") is True:
        txt += (
            "\nAtención: esa fecha cae dentro de la muestra de entrenamiento; el modelo ya vio esos datos, "
            "así que no es una predicción fuera de muestra y no debe leerse como evidencia de capacidad predictiva."
        )
    txt += f"\n(modelo v{clean(p.get('model_version'), 20)}; es una estimación estadística, no una certeza.)"
    return txt


def model_text(m: dict) -> str:
    def block(d: dict) -> str:
        return ", ".join(f"{clean(k, 20)} {num(v, 3)}" for k, v in list(d.items())[:6] if isinstance(v, int | float))

    lines = [
        f"Modelo: {clean(m.get('model_name'), 50)} v{clean(m.get('model_version'), 20)}, entrenado el {clean(m.get('trained_at'), 25)} "
        f"con datos {clean(m.get('trained_on'), 90)}.",
        f"Umbral de decisión: {num(m.get('decision_threshold'))}; {len(m.get('features') or [])} variables.",
        f"Métricas en test: {block(m.get('test_metrics') or {}) or 'n/d'}.",
    ]
    base = m.get("baselines_test_metrics") or {}
    for name, d in list(base.items())[:3]:
        if isinstance(d, dict):
            lines.append(f"Referencia «{clean(name, 30)}»: {block(d) or 'n/d'}.")
    return "\n".join(lines)


def error_text(err) -> str:
    rid = f" (id de petición: {clean(err.request_id, 40)})" if err.request_id else ""
    return f"No pude completar la consulta: {clean(err.message, 300)}{rid}"
