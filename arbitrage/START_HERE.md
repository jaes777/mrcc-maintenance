# Start here

You said you know nothing about coding or share trading. This page assumes
exactly that. Read it before you touch anything else.

---

## The short version

I built you a system that hunts for one specific, real market inefficiency,
measures honestly whether it can be traded profitably, attacks its own findings
to try to disprove them, and paper-trades whatever survives.

**It does not place real orders, and it cannot be made to by accident.**

I could not build the thing you literally asked for — a tool with a guaranteed
minimum 60% win rate — because that thing cannot exist. Nobody can promise a
future win rate: not me, not a bank, not a hedge fund. Anyone who tells you
otherwise is selling something.

So I turned your requirement into something real: **the system measures the win
rate on data it has never seen, and refuses to trade anything that cannot prove
it clears 60%.** That is the strongest honest version of what you asked for.

---

## What the trade actually is

Some companies are listed on two stock exchanges at the same time.

- **BHP** trades in Sydney *and* New York.
- **Shell** trades in London *and* New York.
- **AstraZeneca** trades in London *and* on Nasdaq.

Both listings are claims on the same company. So once you convert the
currencies — and adjust for the fact that one "share" in New York can represent
2, 5, or 8 ordinary shares in the home market — the two prices have to track
each other closely. They are the same thing.

Sometimes they drift apart. Maybe by 0.3%. That gap is the opportunity:

> Buy the listing that is too cheap. Simultaneously sell the listing that is too
> expensive. When the gap closes, you close both and keep the difference.

The appeal is that you do not need to know whether the company will go up or
down. You only need the gap to close. That is what makes it *arbitrage* rather
than speculation.

---

## Why this is much harder than it sounds

Every one of these comes out of your pocket, on **both** legs of **every**
trade:

| Cost | Typical size |
|---|---|
| Broker commission | 1–4 basis points per side |
| The bid/ask spread you cross | 2–6 basis points per side |
| Currency conversion (your broker's cut) | 10–20 basis points |
| UK stamp duty on share purchases | **50 basis points** |
| Hong Kong stamp duty | 13 basis points each way |
| Borrowing shares in order to sell short | ~50 basis points per year |
| Sydney and New York are never open together | real, unavoidable slippage |

A basis point is one hundredth of one percent. Add them up and a "0.3% arbitrage
opportunity" is very often a **losing trade**.

This is the single most important thing the system does: it charges every one of
those costs before it will call anything an opportunity. Run this to see the
arithmetic for a real pair:

```
python3 -m arbtool costs SHEL
```

The bottom line it prints is how far apart the two prices must be before you
make a single penny. Most days, they are closer together than that. Which is why
the correct action, most days, is to do nothing.

---

## How to run it

You need Python 3.9 or newer. Nothing else — no packages to install.

**1. Understand what it does (30 seconds, no internet needed):**

```
cd arbitrage
python3 -m arbtool explain
```

**2. Watch the whole system work, offline, on invented markets:**

```
python3 -m arbtool demo
```

This runs every agent on generated data. No internet needed, no real company
involved. It exists so you can watch the machinery work end to end before you
trust it with anything real. It takes about a minute and writes an HTML report
you can open in a browser.

**3. Check whether real market data is reachable from your machine:**

```
python3 -m arbtool check
```

**4. Do the real thing:**

```
python3 -m arbtool research --verbose
```

This downloads years of real prices for real dual-listed companies, tests
everything, and writes a report. Expect it to take a few minutes the first
time, and to be quick afterwards because prices are cached locally.

**5. See what it would trade today:**

```
python3 -m arbtool signals
```

---

## How to read the answer

The report opens with a verdict in plain words. Then it shows you these numbers.
Learn what they mean, because the first one is the one that will mislead you:

**Win rate.** How often it won. You asked for 60%. **This number on its own is
close to worthless** — see the warning below.

**Win rate, honest floor.** The bottom of a 95% confidence range. If it won 7 of
10 times, the raw win rate reads 70%, but the honest floor is about 39% — the
sample is far too small to claim anything. *This* is the number the system gates
on, not the raw one.

**Profit factor.** Money won divided by money lost. Below 1.0 means it lost money
regardless of how often it won.

**Average per trade.** What one trade earns after every cost. If this is
negative, nothing else matters.

**Worst drawdown.** The biggest fall from a peak. This is the number that decides
whether you would actually have stuck with it, or panicked and quit at the worst
possible moment.

**Could this be luck?** Below 0.05 means probably not.

---

## The warning you most need to hear

> **A high win rate is not the same as making money.**

This kind of strategy naturally produces many small wins and rare, large losses.
You can win 80% of the time and still go broke — win £100 eight times, lose
£1,000 twice, and you are down £1,200 with an 80% win rate.

The system demonstrates this to you directly. One of its adversarial tests runs
the strategy on markets that have been deliberately stripped of any opportunity
— pure noise, nothing to exploit. **The strategy still "wins" 55–60% of the time
on that garbage data.** It loses money doing it, but it wins more often than it
loses.

That is why the gate checks profit factor, profit per trade, and drawdown
alongside your 60%. If you take one thing from this project, take that.

---

## What happens after a strategy passes

Nothing automatic. The verdict is "cleared for **paper** trading" — meaning the
system will now record what it *would* have done, using real live prices and no
money at all.

That is not a formality. A backtest is the strategy grading its own homework.
Forward paper trading is the first genuinely independent evidence you get. Run
it for months, then compare the real record against the backtest. If they
disagree, the backtest was wrong.

**This tool will never place a real order.** If you eventually want that, you
would have to open a broker account that supports both exchanges *and* permits
short selling, write the connection code yourself, and replace my estimated
costs with your broker's real quoted ones. `RISK.md` explains why I left that
step to you rather than doing it for you.

---

## Files, if you are curious

| File | What it is |
|---|---|
| `RISK.md` | The honest list of everything that can go wrong. Read it. |
| `README.md` | Technical documentation of how the system works |
| `arbtool/universe.py` | The company pairs and the cost assumptions — edit these |
| `arbtool/costs.py` | Every fee, tax and spread that gets charged |
| `arbtool/beta.py` | The agents whose job is to prove the strategy wrong |
| `tests/` | 46 tests, mostly checking the engine cannot cheat |

---

## The most likely outcome

Most runs will come back **"not fit to trade."**

That is the system working correctly, not failing. This inefficiency is watched
by firms with faster connections and lower costs than you will ever have. If a
free tool using free daily data found an easy edge in it, the honest conclusion
would be that the tool is broken.

A rejected strategy costs you nothing. An accepted bad one costs you everything
you fund it with. The whole value of what I built is that it is willing to tell
you no.
