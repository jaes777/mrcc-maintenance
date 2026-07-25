# ausform

Australian thoroughbred race analysis and bet evaluation.

It ingests everything knowable about a race — every runner's full form
history, class, speed and sectional figures, barrier, weight, jockey and
trainer records, distance and going preferences, track geometry, weather
and rainfall, and the live betting market — turns it into calibrated win
and place probabilities, and tells you which bets (if any) are worth
making and how much to stake.

---

## Read this first: about the 60% win rate

You asked for a tool that predicts winners at a 60% strike rate. **That is
not achievable, by anyone, with any amount of data or computing power.**
This is not a limitation of the tool. It is a property of horse racing.
I would rather tell you now than sell you a number that quietly loses your
money.

Here is the arithmetic.

In Australian racing the **market favourite wins about 32–35% of races**
(one published analysis: 34.9% across 47,660 races; another: 33.4% across
50,624). Field sizes average 12–13 runners. Now consider a hypothetical
oracle that knows the *true* win probability of every horse — not a good
model, a perfect one — and always names the most likely winner. Its strike
rate is the average of the highest true probability in each race. Since the
betting market is close to efficient at the front end, that number is
roughly the favourite's win rate:

> **A model with perfect knowledge picks the winner of an Australian race
> about 35% of the time.** The remaining 65% is irreducible randomness —
> a bad ride, a slow beginning, interference on the turn, a horse having
> an off day.

For 60% you would need, in every race, one horse whose *genuine* chance is
60% — a $1.67 standout — with the other eleven sharing the remaining 40%.
Australian racing does not look like that and never has.

You *can* hit 60% trivially: bet only odds-on favourites. You will also
lose money steadily, because those bets need to win more often than 1/odds
to profit, and the bookmaker's margin (15–30%) guarantees they do not.
**Strike rate and profitability are close to unrelated.** This is the single
most important thing to understand before betting.

### What is actually achievable

| Metric | Realistic target |
|---|---|
| Top-1 accuracy | **31–36%** — matching or slightly beating the favourite |
| Log-loss vs. the devigged market | Beat it by **0.5–2%** relative. Beating it by 5%+ almost always means a bug |
| Return on tote betting | **Negative**, at 14.5% takeout, unless the edge is very large |
| Return on Betfair, selected bets only | **0% to +5%** on maybe 5–15% of runners |
| Bets needed to *prove* a 5% edge | **~14,400** |

That last row deserves emphasis. Win betting has brutal variance. To
distinguish a real 5% edge from luck at ordinary statistical confidence
takes roughly fourteen thousand bets. At a few qualifying bets per meeting
that is **years**. Any conclusion — yours, mine, or a backtest's — drawn
from a few hundred bets is noise.

Bill Benter, who made a fortune at this in Hong Kong, ran a model whose
combined R² was 0.1396. It explained a small fraction of race outcomes. He
was profitable because he was *slightly* better than the pool, on a subset
of runners, at scale, for years. That is the shape of the opportunity.

**The goal is not to pick more winners than the market. It is to price a
minority of runners slightly better than the market, and bet only those.**
Most races should produce no bet. When this tool says "No bet", it is
working correctly.

---

## What it does

- **Conditional logit (Plackett–Luce) model.** The standard in the racing
  literature, because it models a race as a *competition*: probabilities
  are a softmax over the runners in that race, so they sum to 1 and
  variable field sizes are handled natively. Not a per-horse classifier —
  see `model/conditional_logit.py` for why that approach is wrong and why
  it produces the fake "97% accuracy" figures you'll find online.
- **Benter-style two-stage market blend.** Fits `p ∝ f^α · π^β`, combining
  the fundamental model with the devigged market. This is the highest-value
  component in the system; the market knows things no data feed carries.
- **Point-in-time features, enforced structurally.** A rolling context that
  *raises an exception* if races are processed out of date order.
- **Overround removal** — proportional, power, and Shin's method, with a
  utility to pick the best on your own data by out-of-sample log loss.
- **Correct Australian exotics.** Place rules (8+ starters → 3 places, 5–7
  → 2, ≤4 → none), quinella, exacta, trifecta, first four, quaddie, flexi
  percentages, and the Lo & Bacon-Shone discounted-Harville correction
  (plain Harville systematically overrates a favourite's place chance).
- **Fractional Kelly staking** with per-bet, per-race and per-day caps.
- **Walk-forward backtesting** with log loss, Brier, calibration curves,
  ROI by edge threshold, and — critically — the devigged market baseline
  next to every model number, plus an honest significance test.

---

## Install

```bash
cd horse-racing
python3 -m venv .venv
.venv/bin/pip install -e ".[web,gbm,dev]"
```

## Try it

```bash
.venv/bin/ausform demo
```

This simulates a season, runs a full walk-forward backtest, and analyses a
sample race. It needs no data and no API keys.

```bash
.venv/bin/ausform serve      # web dashboard on http://127.0.0.1:8000
```

---

## Getting real data

This is the hard part, and you should know the landscape before you start.

### Free and legally clean — start here

**Betfair publishes Australian racing data for modellers, free, no login.**

```bash
ausform fetch-betfair --start 2024-01-01 --end 2024-12-31 --db ausform.db
```

Daily starting-price files, one per day, back to the beginning of the
Exchange:
`https://promo.betfair.com/betfairsp/prices/dwbfpricesauswin{DDMMYYYY}.csv`

You get real outcomes and the Betfair Starting Price — an excellent market
consensus — for hundreds of thousands of Australian runners. You do **not**
get barrier, weight, jockey, sectionals or track condition. Treat Betfair
as the outcome-and-price spine and join richer form onto it.

Also free: yearly ANZ thoroughbred extracts at
`https://betfair-datascientists.github.io/data/dataListing/`.

**Weather** comes from Open-Meteo — free, no key. Note the free tier is
non-commercial.

### Grey area — personal use, low request rates

**TAB's public JSON service** (`api.beta.tab.com.au/v1/tab-info-service`)
is what tab.com.au itself uses. No key required. It carries live fields,
fixed and tote odds, barrier, weight, jockey, trainer, track condition and
rail position. But it is undocumented, sits behind bot protection, and
TAB's conditions of use grant only a limited personal-use licence. The
client here throttles to one request per second by default. Do not
redistribute what you pull. For commercial use, apply for the TAB Studio
API.

### Do not scrape

**Racing Australia's terms of use expressly prohibit automated access** —
they specifically bar bots that "search, copy, 'scrape', store and/or
reuse" their material, and reserve the right to pursue legal remedies.
racing.com sits under the same licensing umbrella. Australia has no general
text-and-data-mining exception, so "it's just facts" is a weak defence
here. If you need full form data commercially, licence it — Racing
Australia distributes through authorised wholesalers, and Punting Form
sells a straightforward form-and-results API.

One more regulatory note: using Australian race field information *in
connection with wagering* generally requires approval from each state
principal racing authority and attracts product fees. That binds wagering
operators, not a private tool that never accepts bets — but if you ever
publish tips commercially, get advice.

### Bring your own CSV

```bash
ausform import-csv my_form_data.csv --db ausform.db
```

Required columns: `race_id, date, track, race_number, distance_m, horse_id,
horse_name, finish_position`. Strongly recommended: `field_size, barrier,
weight_kg, jockey, trainer, track_condition, class_level, prize_money,
starting_price, margin_l, race_time_s, last_600m_s`. Full schema in
`src/ausform/data/store.py`.

---

## Then

```bash
ausform backtest --db ausform.db --json report.json   # honest evaluation
ausform train    --db ausform.db --out model.pkl
ausform serve    --db ausform.db --model model.pkl
```

**Backtest before you train, and believe the backtest over your
intuition.** If the model does not beat the devigged market's log loss out
of sample, it has no business placing a bet, however good its winners look.

---

## Where to bet, if you do

| | Tote | Fixed odds | Betfair |
|---|---|---|---|
| Cost | 14.5% of turnover (win); 20–23% on exotics | 15–30% overround | 8% of net winnings (10% NSW/ACT) |
| Price known when you bet | No | Yes | Yes |
| Your bet moves the price | Yes, materially in small pools | No | At depth |
| Winning accounts restricted | No | **Yes, routinely** | No |

For a model-driven approach the exchange is the only structurally viable
venue at scale: commission on *net winnings* is far cheaper than takeout on
*turnover*, and corporate bookmakers close accounts that win. The tool
computes EV net of commission by default.

---

## What independent review found

Two AI reviewers audited this code against numerical references, with
instructions to assume the author was fooling himself. Worth reading,
because it tells you which parts to trust.

**Confirmed sound.** The feature pipeline is leak-free under four
independent probes — including reversing every finishing position in the
target race, and destroying every result *after* it. Feature values changed
by exactly 0.0 in both cases. The conditional-logit gradient matches finite
differences to 1e-8, the exploded Plackett–Luce likelihood matches a
brute-force reference to 0.0, the blend's closed form matches to 1e-17, and
the discounted-Harville fast path matches exhaustive enumeration to 1e-14.
The simulator's private-information term is not recoverable from the full
feature set (out-of-sample R² = −0.012).

**Found broken, now fixed.** The reviewers found bugs I had not, and
several flattered the results:

- Joint Kelly used the wrong inclusion condition, mis-sizing 64% of races.
- Log loss silently *deleted* races where the model called the winner
  impossible — removing its own worst failures from the scorecard.
- The "bets needed to prove this edge" figure was algebraically circular:
  it always said you already had enough evidence precisely when the sample
  had got lucky.
- The simulator handed the model 92% of its hidden ground truth through the
  weight column.
- The simulated market's favourite-longshot gradient ran *backwards*, which
  would have quietly rewarded a model for backing outsiders.

Every one of these has a regression test in `tests/test_audit_regressions.py`.

**The finding that matters most to you.** A reviewer re-ran the same
pipeline across three random seeds. One seed showed "+30% yield,
statistically significant"; the other two were flat to negative. That is
the entire lesson of this project in one experiment: **a good-looking
backtest result is usually luck, and the only defence is to demand far more
evidence than feels necessary.** The tool now reports bootstrap confidence
intervals and warns about multiple testing, precisely so that a lucky run
cannot present itself as an edge.

## Honest limitations

1. **The simulator is not evidence.** `ausform demo` runs on synthetic
   races. It is calibrated against real Australian figures — the favourite
   wins ~32–35%, blind flat betting returns about −29% on longshots rising
   to −11% on short-priced runners, the book runs 118% — and it includes a
   "private information" term the model cannot see, representing what
   stables know and data feeds never carry. It exists to prove the
   *machinery* is correct: that features are point-in-time, the likelihood
   is right, the staking plan does not go broke. **Returns measured on it
   describe the simulator, not Australian racing**, and its market is still
   easier to beat than a real one. Do not use demo yields to set
   expectations; use them to check nothing is obviously broken.
2. **The live adapters could not be tested against live endpoints** from
   the sandbox this was built in — outbound access to tab.com.au,
   betfair.com and open-meteo.com was blocked by network policy. The
   parsers are written against real recorded responses and are covered by
   fixture tests, but the first thing you should do on your own machine is
   run a single request and check the shape.
3. **Sectional times and detailed speed maps are where the real edge is**,
   and they are exactly what the free sources do not carry.
4. **No model survives account restriction.** If you bet fixed odds and
   win, you will be limited.
5. **This tool cannot make gambling profitable for you.** It can tell you
   when the price is wrong, how confident that judgement is, and how little
   to stake. Most of the time the answer is "don't bet".

If gambling stops being something you control, the Australian national
helpline is **1800 858 858**, 24 hours.

---

## Layout

```
src/ausform/
  types.py          domain model, AU track-condition scale
  tracks.py         racecourse registry, coordinates, geometry
  data/
    betfair.py      free public BSP files          <- best free real data
    tab.py          live odds and fields
    weather.py      Open-Meteo
    store.py        SQLite archive + CSV ingest
    simulator.py    synthetic season for testing
  features/
    context.py      point-in-time rolling stats    <- leakage prevention
    build.py        ~105 features
  model/
    conditional_logit.py   Plackett-Luce
    blend.py               Benter two-stage market blend
    gbm.py                 LightGBM alternative
  betting/
    odds.py         devigging, takeout, commission
    exotics.py      Harville + discounted, AU place rules, flexi
    staking.py      Kelly and fractional Kelly
  backtest/
    engine.py       walk-forward
    metrics.py      log loss, calibration, yield, significance
  analyse.py        race -> recommendation
  cli.py, web/      interfaces
```

Run the tests with `.venv/bin/python -m pytest`. The ones in
`tests/test_leakage.py` matter most: they verify that features do not
change when future races are added to the dataset, which is the failure
mode that makes a racing model look brilliant and lose money.
