# arbtool — multi-agent cross-market arbitrage research

Researches statistical arbitrage between dual listings of the same company,
measures whether an edge survives realistic costs and out-of-sample testing,
attacks the result with adversarial agents, and paper-trades what survives.

**Non-technical reader: start with [START_HERE.md](START_HERE.md).**
**Before trusting any number: read [RISK.md](RISK.md).**

Zero required dependencies. Python 3.9+. Everything runs on the standard library.

```bash
cd arbitrage
python3 -m arbtool explain     # what this is, in plain English
python3 -m arbtool demo        # full pipeline, offline, on generated markets
python3 -m arbtool research    # the real thing, on real prices
python3 -m unittest discover -s tests
```

---

## The strategy

Two listings of one company — `BHP` on NYSE and `BHP.AX` on the ASX — are claims
on the same asset. After converting to one currency and dividing by the ordinary
shares each quoted unit represents, their prices must track. The log ratio of
those normalised prices is the *spread*; when it is abnormally wide, short the
rich leg, buy the cheap leg, and exit on convergence.

The edge, if any, is in the spread's mean reversion. The difficulty is entirely
in the costs.

---

## Architecture

```
Scraper ─▶ Normaliser ─▶ PairSelector ─▶ Analyst ─▶ Risk ─▶ Evaluator ─▶ BetaTester ─▶ Execution
```

| Agent | Responsibility |
|---|---|
| **Scraper** | Fetches prices and FX from stooq → Yahoo → cache, with fallback |
| **Normaliser** | Re-derives the share ratio from prices; disqualifies pairs whose configured ratio disagrees with the data |
| **PairSelector** | Correlation, half-life, stationarity, and gap-versus-cost screening, with a written reason for every rejection |
| **Analyst** | Walk-forward: fits 144 model configurations on train windows, scores the chosen one on the following unseen window |
| **Risk** | Concurrency, gross exposure, per-trade stop and daily loss halt applied to the merged trade list |
| **Evaluator** | The promotion gate — six checks, all must pass |
| **BetaTester** | Nine adversarial tests plus an optional independent Claude reviewer |
| **Execution** | Paper broker. `LiveBroker` raises `NotImplementedError` by design |

Every message between agents is recorded and printed in the report, so any
conclusion traces back through the chain that produced it.

---

## The promotion gate

Nothing trades — even on paper — without passing all six:

| Check | Default |
|---|---|
| Out-of-sample trades | ≥ 30 |
| **Win rate, 95% Wilson lower bound** | **≥ 60%** |
| Profit factor | ≥ 1.20 |
| Expectancy per trade | ≥ +1.0 bp |
| Max drawdown | ≤ 15% |
| Binomial p-value vs a coin flip | < 0.05 |

Gating on the *lower bound* is the point: 6 wins from 10 reads as "60%" but has
a lower bound near 39%, and is rejected. `tests/test_arbtool.py` asserts exactly
this case.

## The adversarial agents

| Test | What it proves |
|---|---|
| **Look-ahead audit** | Recomputes every score with future bars removed and requires bit-identical results. A failure invalidates the entire backtest |
| **Cost shock** | Doubles commissions, spreads, FX and borrow |
| **Execution delay** | Fills every order 3 bars late |
| **Data gaps** | Randomly deletes 5% of bars across 5 seeds |
| **Null-market test** | Replaces the spread with a same-magnitude random walk. The strategy must *lose money* on data containing no opportunity |
| **Parameter sensitivity** | Requires ≥ 60% of neighbouring parameter settings to also be profitable — a plateau, not a spike |
| **Worst period** | Reports the worst out-of-sample window with ≥ 5 trades, not the average |
| **Multiple-testing check** | Šidák-corrects the p-value for the size of the parameter search |
| **Tail risk** | Compares the 1st-percentile loss against the average win |

Optionally, `--ai-review` asks Claude (`claude-opus-5`, via the `anthropic`
package) to critique the report independently. It never gates anything, and the
suite runs identically without it.

---

## No-look-ahead guarantees

1. Every model's score at bar *i* uses bars `0..i-1` only; `_zscore` explicitly
   excludes the current bar from its own window, and `_kalman` updates its state
   *after* scoring.
2. A signal at bar *i* is filled at bar *i+1*'s price. Never the same bar.
3. Walk-forward selects parameters on `[t, t+train)` and scores on
   `[t+train+purge, ...)`, with a purge gap between them.
4. Only test-window trades are ever scored.
5. The look-ahead audit re-verifies (1) empirically at 12 checkpoints per pair,
   and `TestModels.test_every_model_is_causal` asserts it for all four models.
6. `TestBetaAgents.test_lookahead_audit_catches_a_cheating_model` registers a
   deliberately non-causal model and asserts the audit fails it — the detector
   itself is tested.

---

## Cost model

Charged per leg, per side, before anything is called an opportunity:

commission · half-spread · UK stamp duty (50 bp on buys) · HK stamp duty (13 bp
both ways) · SEC fee · Taiwan transaction tax · FX conversion spread · stock
borrow accrued daily · a slippage penalty when the two exchanges' sessions never
overlap.

```bash
python3 -m arbtool costs SHEL     # itemised breakeven for a real pair
```

Venue assumptions live in `arbtool/universe.py`. **They are defaults, not your
broker's numbers.** Replace them.

---

## Configuration

```bash
python3 -m arbtool init-config --out my.json
python3 -m arbtool research --config my.json
```

Common flags: `--pairs BHP,SHEL`, `--capital 50000`, `--win-rate 0.65`,
`--provider synthetic`, `--long-only`, `--verbose`, `--ai-review`,
`--report path.html`.

Unknown configuration keys raise rather than being silently ignored, and
`Config.validate()` rejects contradictory settings (exit above entry, live mode
without explicit acknowledgement, and so on) before a run starts.

---

## Data sources

| Provider | Notes |
|---|---|
| `stooq` | Free daily CSV, good international coverage. Tried first |
| `yahoo` | Free JSON chart endpoint. Fallback |
| `synthetic` | Deterministic generated cross-listings. No network. Spread s.d. ≈ 40 bp with regime shifts and jumps |
| `auto` | stooq → yahoo (default) |

All are wrapped in a SQLite cache with stale-data fallback, so a source outage
does not invalidate work already done.

Results from `synthetic` prove the machinery works and say nothing about real
markets. The report and CLI both say so, loudly, whenever it is used.

---

## Module map

| Module | Contents |
|---|---|
| `config.py` | Typed configuration, validation, JSON round-trip |
| `universe.py` | Venues (currency, fees, taxes, sessions) and 15 dual-listed pairs |
| `stats.py` | Wilson interval, half-life, ADF-style statistic, drawdown, binomial tail |
| `costs.py` | The cost model and its plain-English explainer |
| `pairs.py` | Alignment, currency/ratio normalisation, ratio inference, pair screening |
| `models.py` | Four causal signal models (`zscore`, `ratio`, `kalman`, `hybrid`) + grid |
| `backtest.py` | Simulation engine and the walk-forward protocol |
| `portfolio.py` | Trades, portfolio limits, paper broker, live-broker stub |
| `metrics.py` | Win rate with confidence bounds, profit factor, tails, drawdown |
| `beta.py` | Adversarial agents and the optional Claude reviewer |
| `agents.py` | Message bus, the eight agents, orchestrator |
| `report.py` | Self-contained responsive HTML report + JSON |
| `cli.py` | Command line interface |

---

## Tests

```bash
python3 -m unittest discover -s tests -v      # 46 tests, ~2s
```

Covering: statistical correctness, cost accounting, model causality, execution
lag, cost deduction, walk-forward isolation, portfolio limits, gate behaviour
(including the "60% on 10 trades must fail" case and the "80% win rate that
loses money must fail" case), the beta agents, config validation, and a full
end-to-end pipeline run.
