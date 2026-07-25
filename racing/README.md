# Australian Racing Model

A tool that analyses Australian thoroughbred races, estimates each horse's real
chance of winning, and tells you which bets — if any — are priced better than
those chances justify.

### Read this first

**[HONEST_ANSWER.md](HONEST_ANSWER.md)** explains why a 60% win rate is not
achievable by anything, what *is* achievable, and how racing models fool their
own authors. It's written for someone who doesn't code and doesn't follow
racing. Please read it before betting money.

The short version: the market favourite wins about 31% of races and the market
is well calibrated. A good model's top pick wins 30–36% — about the same. Money
comes from finding **mispriced** horses, not from picking more winners.

---

## Setup

You need Python 3.10 or newer. In a terminal, from this folder:

```bash
pip install -r requirements.txt
python run.py check
```

`check` verifies the packages installed, tests that it can reach the data
sources, and runs the arithmetic tests. If anything fails it will say what.

---

## Try it without any data

```bash
python run.py demo
```

This simulates several thousand races, builds the model, tests it properly, and
prints the results — including an example race with bet recommendations. Takes
a few minutes.

**The races are invented.** The demo shows the machinery works. It says nothing
about real racing.

---

## Use it on real Australian racing

### 1. Download the data

```bash
python run.py fetch --days 1460 --weather
```

Downloads about four years of Australian thoroughbred results and prices from
Betfair's public data hub, and attaches rainfall history from Open-Meteo.

Both sources are free, need no account, and are used within their terms.
Betfair publishes these files specifically for people building models.

This takes a while (it's a lot of data) and caches to `data/raw/` so you only
do it once.

### 2. Test it honestly

```bash
python run.py backtest --commission 0.08
```

This is the step that matters. It trains on the past, predicts races it has
never seen, rolls forward, and repeats — exactly how you'd have used it in real
life. It reports:

- whether the model beats the market's own accuracy (**the key number** — if it
  doesn't, nothing else matters)
- how often its top pick won, next to how often the favourite won
- return on money staked, **and whether that's distinguishable from luck**
- calibration: when it says 20%, does it win 20%?
- return broken down by price bracket
- warnings if anything looks too good to be true

Use `--commission 0.08` for Betfair, or `0` for a bookmaker. Add `--no-market`
to see how the model does on form alone, ignoring the odds — that's the honest
test of whether it knows anything the market doesn't.

### 3. Build a model

```bash
python run.py train
```

Saves to `models/model.pkl` and prints which factors it leaned on most.

### 4. Analyse an upcoming race

Make a CSV with one row per runner:

```csv
race_id,race_datetime,track,runner_id,horse,tab_number,barrier,weight_kg,jockey,trainer,distance_m,track_condition,race_class,odds_decimal
R1,2026-08-01 14:30,Flemington,H001,Fast Lad,1,3,58.5,J Smith,T Jones,1400,Good 4,BM78,4.20
R1,2026-08-01 14:30,Flemington,H002,Slow Coach,2,7,56.0,A Brown,B White,1400,Good 4,BM78,7.50
```

Only `race_id`, `race_datetime`, `track`, `runner_id` and `horse` are required.
Everything else improves the estimate — `odds_decimal` most of all.

```bash
python run.py predict my_race.csv --bankroll 500
```

You get each horse's win and place probability, its fair price, and any bets
worth making. **Usually there won't be any.** That's correct behaviour.

---

## What it looks at

About 60 factors per runner:

**Form** — career starts, win and place rates, recent finishing positions,
beaten margins, momentum (are the last two runs better than the three before?)

**Fitness** — days since last run, first-up from a spell, second-up

**Suitability** — record at this distance, this track, this going, this class;
whether the horse is a wet-tracker; change of distance from last start

**Conditions** — track rating on the Australian 1–10 scale, rainfall over the
last 1/3/7/14 days, temperature, wind

**The race** — barrier (relative to field size, and weighted more over sprint
distances), weight carried relative to the field and versus last start, field
size, prize money, class movement

**People** — jockey and trainer strike rates, their record at this track, and
the jockey-trainer combination — all statistically shrunk, because a jockey
with 2 wins from 3 rides does not have a 67% strike rate

**The market** — the de-vigged price, its rank, and how the horse's price has
moved since its last start

### Traps it avoids

**Weight.** Raw weight correlates *positively* with winning, because better
horses are assigned more weight. A naive model learns "heavier is better". This
one uses weight relative to the field and change since last start.

**Jockey and trainer statistics.** Mostly a proxy for horse quality — good
jockeys ride good horses. Shrunk toward the population average so small samples
don't dominate.

**Small samples generally.** Every rate is shrunk. A horse with one run on
heavy going doesn't get a 100% wet-track record.

---

## How the model works

Three stages:

1. **A gradient-boosted tree** scores each runner from its form. Trees handle
   racing's messy, interaction-heavy data far better than a linear model and
   cope with missing values, which matters because racing data is full of holes.

2. **Softmax within the race** turns those scores into probabilities that sum
   to 1 across the field. This is what makes it a *racing* model: a horse's
   chance depends entirely on who else is in the race.

3. **A blend with the market price**, following Bill Benter's 1994 method:

   ```
   final probability ∝ (model probability)^α × (market probability)^β
   ```

   with α and β fitted on held-out races. The market aggregates information the
   model can't see — stable gossip, trackwork, late money. Benter found the
   combination beat either part alone, and that his fundamental model *by
   itself* was barely better than just reading the odds.

You will usually see β larger than α. That means the market knows more than the
form does. It's the normal, healthy result.

---

## Bet types

Win, place, each-way, quinella, exacta, trifecta, first four, boxed and standout
tickets, flexi betting, and same-race multis.

Place terms follow the Australian rules and are computed from the **final**
starter count after scratchings — 8 or more runners pays 3 places, 5 to 7 pays
2, under 5 has no place market. A 9-horse race that loses two runners becomes a
2-place race, and getting that wrong quietly corrupts every place calculation.

Finishing-order probabilities use **discounted Harville**. Plain Harville — the
standard formula — systematically overestimates a favourite's chance of running
2nd or 3rd, because it assumes a beaten favourite was still nearly the best
horse, when really it usually just ran badly. Pricing trifectas off raw Harville
is the most common way model-driven exotic betting loses money. The discount
corrects it, and `exotics.fit_lambdas` refits the correction on your own data.

Same-race multis are priced by simulation, never by multiplying the legs
together. Legs in one race compete for overlapping finishing slots, so they're
*negatively* correlated — multiplying overstates your chances. In a 10-horse
race with every runner at 10%, two horses each have a 30% chance of running top
3, but the joint chance is 6.7%, not 9%.

### Staking

Quarter Kelly by default, capped at 2% of bankroll per bet and 5% per race.

Kelly maximises long-run growth but is brutally aggressive, and it assumes you
know the true probability — you don't, you have a noisy estimate, and it's
biased optimistic because you bet where the model is most confident.

For scale: a 5% edge at $5.00 justifies a **1.25%** bet at full Kelly, and
**0.31%** at quarter Kelly. Anyone staking 5–10% per bet isn't doing Kelly.

---

## Where the data comes from

| Source | Cost | What it gives you |
|---|---|---|
| [Betfair ANZ Thoroughbreds](https://betfair-datascientists.github.io/data/dataListing/) | Free | AU/NZ races, prices, Betfair Starting Price, winners |
| [Betfair SP daily files](https://promo.betfair.com/betfairsp/prices) | Free | Daily settled results, to stay current |
| [Open-Meteo](https://open-meteo.com) | Free (non-commercial) | Rainfall and weather per racecourse |
| Betfair Exchange API | Free | Live prices, barrier, jockey, trainer, weight |
| [Punting Form](https://puntingform.com.au) | ~A$59/mo | Real form history and sectional times |

### Two gaps in the free data

**Finishing positions.** Betfair's files record only who *won*, not who ran
second or third. So form features based on finishing position are mostly empty.
The tool detects this and tells you.

**Track condition.** Not in the free data at all. The tool estimates the going
from rainfall history, which is a reasonable proxy — but it's an estimate, and
it can't see irrigation or drainage. Real ratings need a licensed source.

Both gaps are why Punting Form is the one paid source genuinely worth it.

### What this tool will not do

It does not scrape racingaustralia.horse, racing.com, punters.com.au or
racenet.com.au. Racing Australia's terms of use prohibit automated collection by
name and the others take the same position. Those scrapers are easy to write.
That isn't the point.

---

## Project layout

```
run.py                    The command line — start here
HONEST_ANSWER.md          Why 60% is impossible; read this
src/ausrace/
  schema.py               Canonical data format, validation
  features.py             ~60 features, built leak-free
  model.py                LightGBM + softmax + Benter market blend
  exotics.py              Harville / discounted Harville / Stern
  betting.py              De-vigging, expected value, Kelly staking
  tote.py                 Tote takeout, breakage, flexi betting
  evaluate.py             Metrics, calibration, statistical significance
  backtest.py             Walk-forward testing, bankroll simulation
  recommend.py            The bet recommendation engine
  sources/
    betfair_hub.py        Free Betfair data
    weather.py            Open-Meteo + track condition estimation
    synthetic.py          Race simulator, for testing only
tests/                    Arithmetic tests
scripts/                  Validation harnesses
```

---

## Responsible gambling

This tool gives statistical estimates. It is not financial advice, guarantees
nothing, and can lose you money. Only stake what you can afford to lose
entirely.

- **National Gambling Helpline: 1800 858 858** — free, confidential, 24/7.
  [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au)
- **BetStop** — self-exclude from every licensed Australian wagering provider
  in one process. [betstop.gov.au](https://www.betstop.gov.au) or 1800 238 786
- **Lifeline: 13 11 14**

18+.
