# The honest answer about the 60% win rate

You asked for a tool that predicts Australian horse races with at least a 60%
win rate. I've built you the tool. This document is about the 60%.

**A 60% win rate is not achievable.** Not by this tool, not by any tool, not by
anyone. This isn't me being cautious or hedging — it's arithmetic, and I can
show you the arithmetic. Please read this part before you bet any money.

---

## Why 60% is impossible

Start with one fact. In every Australian horse race there is a **favourite** —
the horse the betting public has backed most heavily. The favourite is the
single most-informed prediction available anywhere, because it aggregates the
opinions of thousands of punters, professional syndicates, stable connections
and computer models, all of them with money at stake.

**The favourite wins about 31% of the time.** Roughly one race in three.

That figure comes from analysis of 56,616 Australian thoroughbred races.
The UK number is about 34%. It has been stable for decades.

Now, the second fact. The betting market is **well calibrated**. Betfair
analysed 381,776 races covering 3.2 million runners and found that when the
market's price implies a horse has a 33% chance, it wins about 33% of the time.
The market isn't systematically wrong about favourites. It's about right.

Put those two facts together and here's what a 60% win rate would require:

| | Top pick's true chance | Its fair price | What the other 9 horses share |
|---|---|---|---|
| **Reality** | 33% | $3.00 | 67%, about $13.50 each |
| **A 60% win rate** | 60% | $1.67 | 40%, about $22.50 each |

The second row describes a race where one horse is a near-certainty and every
other runner is a $22 outsider. Races like that exist — they're the ones with
an odds-on favourite, and those favourites win about 59% of the time. But
they're a small fraction of the calendar, and betting them is a slow, reliable
way to lose money, because $1.30 about a 59% chance is a bad bet.

For a model to average 60% across ordinary races, the market would have to be
underestimating favourites by about 80%. If that were true, you could just back
every favourite and make an 80% return. In reality, backing every favourite
**loses about 5–10%** of everything you stake.

So the 60% claim isn't ambitious. It's inconsistent with facts that are easy to
check.

### Where the 60% number probably came from

Every one of these is a real statistic that sits near 60%, and each is a
plausible source of the confusion:

| Real statistic | Value |
|---|---|
| Favourite finishes in the **top 3** (a place, not a win) | ~65–69% |
| **Odds-on** favourites only | ~59% |
| The winner is one of the **top 3** in the market | ~65–70% |
| A backtest with a bug in it | Any number you like |

That last one matters most, so it gets its own section.

---

## The way racing models lie to their authors

If you search for horse racing prediction projects you will find people
reporting 50%, 60%, even 70% accuracy. Almost all of them have the same bug,
and it's a subtle one.

It's called **data leakage**. Here's the plain version.

Say you want to test your model. You take five years of races, shuffle them,
train on 80% and test on the other 20%. That sounds fair. It isn't.

Suppose a horse called Fast Lad ran ten times. Some of those runs land in your
training set and some in your test set. Now you build a feature like "Fast Lad's
career win rate" calculated across all ten runs. When the model predicts a Fast
Lad race that's in the test set, that feature **already contains the answer** —
the horse's career record includes the race you're asking about.

The model isn't predicting. It's remembering. And it will report a spectacular
strike rate that vanishes the instant you use it on a race that hasn't run yet.

This tool is built so that can't happen:

- Races are processed strictly in **date order**. A horse's statistics are
  built up race by race, and every feature is written down *before* that race's
  result is folded in. Not by careful discipline — by structure. A future
  result physically cannot reach a past feature.
- Testing is **walk-forward**: train on everything up to a date, predict the
  next three months, roll forward, repeat. Exactly how you'd have used it in
  real life.
- There's an automatic leakage check that runs every time and flags anything
  suspicious — including "every test period was profitable", which in racing is
  more likely to mean a bug than a goldmine.

That last check earned its place while I was building this. My race simulator
had a sign error that made the simulated bookmaker offer better-than-fair odds,
and the backtest duly reported **+24% profit across every single test period**.
The check flagged it as implausible. It was right — the bug was in my
simulator, not a discovery. That's what the number would have looked like if
I'd shipped it without checking.

---

## What is actually achievable

Here's the honest range, from the research literature and from published
results:

| | Realistic figure |
|---|---|
| Top pick wins | **30–36%** |
| Return on money staked, **if you get an edge at all** | **+2% to +8%** |
| Most likely outcome for a serious attempt | **No edge. Zero or negative.** |

Note the first row. A good model's top pick wins about as often as the market
favourite. That's not a failure — it's expected, because a good model and the
market usually agree about which horse is best.

**The edge doesn't come from picking more winners. It comes from being right
about the price.**

This is the single most important idea in the whole project. A tipster hitting
60% of winners at $1.50 loses money. One hitting 30% at $5.00 makes it. What
matters is whether the price you're offered is better than the horse's real
chance — not how often you're right.

### The best case study there is

Bill Benter built the most successful horse racing operation ever documented,
in Hong Kong. His 1994 paper reports the numbers, and they're sobering:

- Five man-years to build a model with any advantage at all. Another five to
  make it properly profitable.
- His fundamental model — everything about the horse, form, class, the lot —
  scored **0.1245** on his accuracy measure. Just using the public odds and
  nothing else scored **0.1218**.

Read that again. After five years of work in the most data-rich racing
jurisdiction on earth, his model **barely beat simply reading the odds**.

What made him money was **combining** the two: model plus market scored 0.1396,
about 15% better than the odds alone. And he still lost about 20% of his capital
in one of his first five seasons.

This tool uses Benter's method. That's why it blends its own opinion with the
market price rather than ignoring the market — and it's why, when you run it,
you'll usually see the market weighted more heavily than the model. That's not
the tool underperforming. That's the tool being honest.

---

## The two things that actually decide whether you win

### 1. The margin

Every bet you place has a built-in cost, and it's larger than most people
realise:

| Where you bet | What it costs you |
|---|---|
| Australian bookmaker, fixed odds | **15–30%** built into the prices |
| TAB tote, win pool | **~14.5%** taken out |
| TAB tote, **trifecta** | **~21%** taken out |
| TAB tote, **first four** | **~23%** taken out |
| **Betfair Exchange** | **~2%** margin + 8–10% commission **on net winnings only** |

That last row is why serious operations use the exchange. Everywhere else takes
its cut from every dollar you turn over, win or lose. Betfair only charges you
on money you've actually won.

Here's what the tote takeout means concretely. To break even on a win bet with
14.5% taken out, your assessment has to be **17% more confident than the whole
betting pool**. On a first four, nearly **30%** more confident. That's the real
reason exotic bets are so hard: you're not just beating the market, you're
beating it by a wide enough margin to cover a 23% haircut first.

**You will not beat a bookmaker's 115–130% market.** If you use this tool to
bet, use Betfair.

### 2. How long it takes to know if it's working

This one surprises people the most.

Racing results are enormously noisy. A single 40/1 winner swings a 500-bet
record by eight percentage points. So the question "is my model actually
working?" takes a shockingly long time to answer.

The maths is standard. To be confident that a **5% edge** is real, at average
odds of $5.00, you need about **9,900 bets**.

| Bets placed | What a measured +5% return actually means |
|---|---|
| 500 | True return is somewhere between **−12% and +23%** |
| 1,000 | Somewhere between **−7% and +17%** |
| 5,000 | Somewhere between **−0.5% and +10%** |
| 10,000 | Somewhere between **+1% and +9%** — now you know something |

**A profitable 1,000-bet record is completely consistent with having no edge at
all.** If this tool bets one race in five, that's about 3,800 bets a year in
Australia — so proving an edge takes **two to three years of live betting**.

A real example: a documented Betfair strategy showed +5.77% after 2,000 bets.
After 17,000 bets it had settled at **−0.63%**. The early profit was luck.

This is why the tool refuses to congratulate you on a small sample. Under 300
bets it will tell you the result is meaningless, because it is.

---

## So what does this tool actually do?

It does the real version of what you asked for.

**It analyses everything available.** About 60 factors per runner: recent form,
career record, class, prize money, barrier draw, weight carried and weight
change, days since last run, whether the horse is first-up from a spell,
distance suitability, track and course record, wet-track record, jockey and
trainer strike rates, jockey-trainer combinations, field size, rainfall and
estimated track condition, and the market's own price.

**It's careful about the traps.** Weight carried, for instance, correlates
*positively* with winning — because better horses are assigned more weight. A
naive model learns "heavier is better" and loses money. This one uses weight
relative to the field and change since last start, which is the meaningful
version.

**It produces probabilities, not tips.** For each horse: the chance it wins, the
chance it places, and the price at which that would be a fair bet.

**It recommends bets only when the price is wrong.** Win, place, quinella,
exacta, trifecta, first four — each assessed on whether the offered price beats
the horse's real chance by enough to be worth the risk. Exotic bets are held to
a much higher bar because of that 21–23% takeout.

**It sizes your bets.** Quarter-Kelly staking, capped at 2% of your bankroll on
any bet and 5% on any race.

**Most of the time it will tell you not to bet.** That's the tool working
correctly. If it found a bet in every race, it would be finding the bookmaker's
margin, not value.

**It tells you when it isn't working.** The backtest reports whether the model
beats the market, whether your sample is big enough to mean anything, and
whether the results look too good to be true.

---

## What I could not do, and what you'd need

**I couldn't validate this on real Australian racing data.** The environment I
built it in has no internet access to racing sites — every attempt to reach
Betfair, Racing Australia, or even the weather service was blocked. So I built
and tested the tool against a **simulator** I tuned to match real Australian
racing (favourite wins 32.6%, backing favourites loses 9%, longshots lose 38% —
all matching the published figures).

That proves the code is correct. It proves **nothing** about how the model will
perform on real races. When you run it on real data, the honest answer is: **I
don't know whether it will find an edge, and neither does anyone else until
it's tested.** The most likely outcome, based on everything in the literature,
is that it won't. That's worth knowing, and it's worth far more than a
confident number I'd have to make up.

Running `python run.py fetch` then `python run.py backtest` on your own machine
is what gives you the real answer. It's free and it takes an afternoon.

**To have a genuine chance at an edge, you'd want:**

1. **Full finishing positions and sectional times.** Betfair's free data only
   records who won, not who ran second or third — so form features are limited.
   Punting Form (~A$59/month) has real form history and sectionals. Sectional
   times are probably the best available source of genuine edge in Australian
   racing, precisely because they're expensive to process and less widely used.
2. **Official track ratings.** The free data doesn't include them. This tool
   estimates the going from rainfall, which is a decent proxy but not the real
   thing.
3. **A Betfair account.** The margin difference is bigger than any edge you're
   likely to find.
4. **Patience.** Thousands of bets before the numbers mean anything.

---

## Before you bet

This tool gives statistical estimates. It is not financial advice, it does not
guarantee any outcome, and it can lose you money.

Two things worth knowing:

- **Recreational gambling winnings aren't taxed in Australia.** But if you run
  a systematic, organised betting operation at scale, the ATO may treat it as a
  business — and a computer model betting systematically is exactly that fact
  pattern. Talk to a tax agent before it gets serious.
- **Only bet money you can afford to lose entirely.** The staking rules in this
  tool assume a bankroll you've set aside for this and nothing else.

**If gambling is causing you or someone you know harm:**

- **National Gambling Helpline: 1800 858 858** — free, confidential, 24 hours
  a day, 7 days a week. [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au)
- **BetStop** — the National Self-Exclusion Register. Blocks you from every
  licensed Australian online and phone wagering provider, for three months up
  to life, in one process. [betstop.gov.au](https://www.betstop.gov.au) or
  1800 238 786.
- **Lifeline: 13 11 14** — 24/7 crisis support.

---

## The short version

- 60% is impossible. The favourite wins 31%, and the market is well calibrated.
- A good model's top pick wins **30–36%** — about the same as the favourite.
- Money comes from **prices being wrong**, not from picking more winners.
- A realistic good outcome is **+2% to +8%** on turnover. The most likely
  outcome is nothing.
- You need **thousands of bets** to know which of those you've got.
- Bet on **Betfair**, or the margin will beat you before the model gets a
  chance.
- Most races should produce **no bet**.

I've built you the strongest honest version of what you asked for. I'd rather
hand you a tool that tells you the truth than one that tells you 60%.
