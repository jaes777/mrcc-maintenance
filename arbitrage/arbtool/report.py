"""Turns a research run into a single self-contained HTML page.

Written for someone who does not read code and does not trade: the verdict comes
first in plain words, the reasoning follows, and the numbers that could mislead
are labelled as such.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import List

from .agents import ResearchOutcome

CSS = """
:root{--bg:#0f1218;--card:#171b24;--line:#262c38;--fg:#e6e9ef;--dim:#98a1b3;
--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--accent:#60a5fa}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ec;
--fg:#1a1d24;--dim:#5c6472;--ok:#15803d;--warn:#b45309;--bad:#b91c1c;--accent:#1d4ed8}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1rem;background:var(--bg);color:var(--fg);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:1.9rem;margin:0 0 .25rem}
h2{font-size:1.25rem;margin:2.5rem 0 .75rem;padding-bottom:.4rem;
border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.5rem 0 .5rem}
.sub{color:var(--dim);margin:0 0 2rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:1.25rem;margin:1rem 0}
.verdict{border-left:5px solid var(--accent);padding:1.25rem 1.5rem}
.verdict.no{border-left-color:var(--bad)}
.verdict.yes{border-left-color:var(--ok)}
.verdict h2{border:0;margin:0 0 .5rem;font-size:1.4rem}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.75rem}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem}
.stat .k{color:var(--dim);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
.stat .v{font-size:1.5rem;font-weight:600;margin-top:.2rem}
.stat .n{color:var(--dim);font-size:.8rem;margin-top:.2rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.9rem;min-width:520px}
th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:.78rem;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:.1rem .5rem;border-radius:99px;font-size:.75rem;
font-weight:600;border:1px solid currentColor}
ul{padding-left:1.2rem}
pre{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1rem;overflow-x:auto;font-size:.8rem;line-height:1.45}
.warning{background:rgba(251,191,36,.12);border:1px solid var(--warn);
border-radius:10px;padding:1rem 1.25rem;margin:1.25rem 0}
footer{color:var(--dim);font-size:.82rem;margin-top:3rem;border-top:1px solid var(--line);
padding-top:1rem}
"""


def _esc(text) -> str:
    return html.escape(str(text))


def _stat(label: str, value: str, note: str = "", cls: str = "") -> str:
    return (f'<div class="stat"><div class="k">{_esc(label)}</div>'
            f'<div class="v {cls}">{_esc(value)}</div>'
            f'<div class="n">{_esc(note)}</div></div>')


def build_html(outcome: ResearchOutcome) -> str:
    m = outcome.gate.metrics
    gate = outcome.config.gate
    deployable = outcome.deployable

    verdict_class = "yes" if deployable else "no"
    if deployable:
        headline = "This strategy cleared every check"
        body = (
            f"Out of sample, it won {m.win_rate:.1%} of its {m.trades} trades, and the "
            f"statistically honest lower bound on that figure is "
            f"{m.win_rate_lower:.1%} — above your {gate.min_win_rate:.0%} requirement. "
            f"It also survived every adversarial test. That earns it a place in "
            f"<strong>paper trading</strong>, and nothing more: the next step is to run "
            f"it on live prices with no money at stake and see whether the real world "
            f"agrees."
        )
    else:
        first = outcome.gate.reasons[0] if outcome.gate.reasons else \
            "it did not survive adversarial testing"
        headline = "This strategy is not fit to trade"
        body = (
            f"It failed at least one required check — {_esc(first)}. That is the system "
            f"doing its job. A rejected strategy costs you nothing; an accepted bad one "
            f"costs you everything you fund it with."
        )

    parts: List[str] = [
        '<div class="wrap">',
        "<h1>Cross-market arbitrage research report</h1>",
        f'<p class="sub">Generated {_esc(datetime.now().strftime("%d %B %Y, %H:%M"))} '
        f'&middot; data source: {_esc(outcome.provider_name)} '
        f'&middot; {len(outcome.series)} pairs examined '
        f'&middot; run took {outcome.elapsed_seconds:.1f}s</p>',
        f'<div class="card verdict {verdict_class}"><h2>{_esc(headline)}</h2>'
        f"<p>{body}</p></div>",
    ]

    if outcome.used_synthetic:
        parts.append(
            '<div class="warning"><strong>These results are from generated data, '
            'not real markets.</strong> They prove the machinery works end to end. '
            'They say nothing whatsoever about whether this would make money. '
            'Re-run without <code>--provider synthetic</code> on a machine with '
            'internet access to get results about real companies.</div>'
        )

    # --- headline numbers ---------------------------------------------------
    parts.append("<h2>The numbers that matter</h2>")
    parts.append('<div class="grid">')
    parts.append(_stat("Win rate (measured)", f"{m.win_rate:.1%}",
                       f"{m.wins} wins of {m.trades} trades"))
    parts.append(_stat("Win rate (honest floor)", f"{m.win_rate_lower:.1%}",
                       "95% confidence lower bound — this is what is gated on",
                       "ok" if m.win_rate_lower >= gate.min_win_rate else "bad"))
    parts.append(_stat("Profit factor", f"{m.profit_factor:.2f}",
                       "money won divided by money lost",
                       "ok" if m.profit_factor >= gate.min_profit_factor else "bad"))
    parts.append(_stat("Average per trade", f"{m.expectancy_bps:+.1f} bp",
                       "after all costs",
                       "ok" if m.expectancy_bps > 0 else "bad"))
    parts.append(_stat("Worst drawdown", f"{m.max_drawdown_pct:.1f}%",
                       "largest peak-to-trough fall",
                       "ok" if m.max_drawdown_pct <= gate.max_drawdown_pct else "bad"))
    parts.append(_stat("Could this be luck?", f"p = {m.luck_pvalue:.4f}",
                       "below 0.05 means probably not",
                       "ok" if m.luck_pvalue < 0.05 else "warn"))
    parts.append(_stat("Net result", f"{m.net_pnl:,.0f}",
                       f"on {outcome.config.risk.capital:,.0f} of capital",
                       "ok" if m.net_pnl > 0 else "bad"))
    parts.append(_stat("Costs paid", f"{m.total_costs:,.0f}",
                       f"{m.cost_share_of_gross:.0f}% of gross profit"))
    parts.append("</div>")

    parts.append(
        '<div class="warning"><strong>A high win rate is not the same as making '
        'money.</strong> Strategies of this type win often and lose rarely but '
        'badly. That is why the report shows profit factor, average profit per '
        'trade and worst drawdown next to the win rate — if any of those is bad, '
        'the win rate is a trap.</div>'
    )

    # --- gate ---------------------------------------------------------------
    parts.append("<h2>Every check, and whether it passed</h2>")
    parts.append('<div class="scroll"><table><tr><th>Check</th><th>Result</th>'
                 '<th>Detail</th></tr>')
    for name, (ok, detail) in outcome.gate.checks.items():
        cls, label = ("ok", "PASS") if ok else ("bad", "FAIL")
        parts.append(f'<tr><td>{_esc(name)}</td>'
                     f'<td><span class="pill {cls}">{label}</span></td>'
                     f"<td>{_esc(detail)}</td></tr>")
    parts.append("</table></div>")

    # --- pairs --------------------------------------------------------------
    parts.append("<h2>Which markets were considered</h2>")
    parts.append('<div class="scroll"><table><tr><th>Pair</th><th>Verdict</th>'
                 '<th class="num">Days</th><th class="num">Correlation</th>'
                 '<th class="num">Gap closes in</th><th class="num">Gap vs cost</th>'
                 "<th>Why not</th></tr>")
    for q in outcome.assessments:
        cls, label = ("ok", "tradeable") if q.tradeable else ("bad", "rejected")
        hl = "n/a" if q.spread_halflife is None else f"{q.spread_halflife:.0f} days"
        parts.append(
            f'<tr><td><strong>{_esc(q.pair_id)}</strong></td>'
            f'<td><span class="pill {cls}">{label}</span></td>'
            f'<td class="num">{q.bars}</td>'
            f'<td class="num">{q.return_correlation:+.2f}</td>'
            f'<td class="num">{_esc(hl)}</td>'
            f'<td class="num">{q.opportunity_ratio:.2f}x</td>'
            f"<td>{_esc('; '.join(q.reasons) if q.reasons else '—')}</td></tr>"
        )
    parts.append("</table></div>")

    # --- walk forward -------------------------------------------------------
    if outcome.walk_forward:
        parts.append("<h2>Fitted on the past, scored on the future</h2>")
        parts.append(
            "<p>Each row is one window of history. The settings were chosen using "
            "only the earlier data and then applied, untouched, to the later data. "
            "Only the <em>unseen</em> column counts. A large drop between the two "
            "columns means the settings were memorising the past.</p>"
        )
        parts.append('<div class="scroll"><table><tr><th>Pair</th><th>Unseen period</th>'
                     "<th>Model chosen</th><th class='num'>Fitted win rate</th>"
                     "<th class='num'>Unseen win rate</th>"
                     "<th class='num'>Unseen trades</th></tr>")
        for pair_id, wf in outcome.walk_forward.items():
            for fold in wf.folds:
                if not fold.chosen:
                    continue
                train_wr = f"{fold.train_metrics.win_rate:.0%}" if fold.train_metrics else "—"
                test_wr = f"{fold.test_metrics.win_rate:.0%}" if fold.test_metrics else "—"
                trades = fold.test_metrics.trades if fold.test_metrics else 0
                parts.append(
                    f"<tr><td>{_esc(pair_id)}</td>"
                    f"<td>{_esc(fold.test_start)} to {_esc(fold.test_end)}</td>"
                    f"<td>{_esc(fold.chosen.name)} "
                    f"(lookback {fold.chosen.lookback}, entry {fold.chosen.entry_z})</td>"
                    f'<td class="num">{train_wr}</td>'
                    f'<td class="num">{test_wr}</td>'
                    f'<td class="num">{trades}</td></tr>'
                )
        parts.append("</table></div>")

    # --- beta ---------------------------------------------------------------
    if outcome.beta:
        parts.append("<h2>Independent agents trying to break it</h2>")
        for pair_id, report in outcome.beta.items():
            parts.append(f"<h3>{_esc(pair_id)}</h3>")
            parts.append('<div class="scroll"><table><tr><th>Test</th><th>Result</th>'
                         "<th>What it found</th></tr>")
            for f in report.findings:
                if f.passed:
                    cls, label = "ok", "PASS"
                elif f.severity == "critical":
                    cls, label = "bad", "FAIL"
                else:
                    cls, label = "warn", "WARN"
                parts.append(f'<tr><td>{_esc(f.name)}</td>'
                             f'<td><span class="pill {cls}">{label}</span></td>'
                             f"<td>{_esc(f.headline)}</td></tr>")
            parts.append("</table></div>")

    if outcome.reviewer_notes:
        parts.append("<h2>Second opinion from an independent AI reviewer</h2>")
        parts.append(f"<pre>{_esc(outcome.reviewer_notes)}</pre>")

    # --- signals ------------------------------------------------------------
    parts.append("<h2>What the strategy would do right now</h2>")
    if outcome.signals:
        parts.append('<div class="scroll"><table><tr><th>Pair</th><th>Date</th>'
                     "<th class='num'>Gap</th><th class='num'>How unusual</th>"
                     "<th>Action</th></tr>")
        for s in outcome.signals:
            action = "; ".join(f"{o['side']} {o['quantity']:.0f} of leg {o['leg']}"
                               for o in s["orders"])
            parts.append(f"<tr><td>{_esc(s['pair'])}</td><td>{_esc(s['date'])}</td>"
                         f"<td class='num'>{s['gap_bps']:+.0f} bp</td>"
                         f"<td class='num'>{s['z']:+.2f} sd</td>"
                         f"<td>{_esc(action)}</td></tr>")
        parts.append("</table></div>")
        parts.append("<p><strong>These are paper orders.</strong> Nothing has been "
                     "sent to any broker, and this tool cannot send them.</p>")
    else:
        parts.append("<p>No trade today. Either no pair is approved for trading, or "
                     "no gap is currently wide enough to be worth the cost of "
                     "trading it. Doing nothing is the correct action far more often "
                     "than not.</p>")

    # --- audit --------------------------------------------------------------
    parts.append("<h2>Full agent transcript</h2>")
    parts.append("<p>Every decision every agent made, in order. Nothing in this "
                 "report comes from anywhere else.</p>")
    parts.append(f"<pre>{_esc(outcome.transcript)}</pre>")

    parts.append(
        "<footer>Research and paper-trading tool. It does not place real orders and "
        "is not investment advice. Past performance — especially simulated past "
        "performance — does not predict future results.</footer></div>"
    )
    return "\n".join(parts)


def write_report(outcome: ResearchOutcome, path: str) -> str:
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Arbitrage research report</title>"
        f"<style>{CSS}</style></head><body>{build_html(outcome)}</body></html>"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return path


def write_json(outcome: ResearchOutcome, path: str) -> str:
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": outcome.provider_name,
        "synthetic": outcome.used_synthetic,
        "deployable_to_paper": outcome.deployable,
        "gate_passed": outcome.gate.passed,
        "gate_checks": {k: {"passed": v[0], "detail": v[1]}
                        for k, v in outcome.gate.checks.items()},
        "metrics": outcome.gate.metrics.to_dict(),
        "pairs": [q.__dict__ for q in outcome.assessments],
        "approved_pairs": outcome.approved_pairs,
        "signals": outcome.signals,
        "beta": {pid: [f.__dict__ for f in rep.findings]
                 for pid, rep in outcome.beta.items()},
        "reviewer_notes": outcome.reviewer_notes,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return path
