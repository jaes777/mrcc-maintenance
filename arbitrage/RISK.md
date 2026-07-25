# Risks and limitations

Everything below is a real way this system, or the strategy it researches, can
lose you money. None of it is boilerplate.

---

## 1. The 60% win rate is measured, never promised

No system can guarantee a future win rate. What this one does is refuse to
approve anything that cannot demonstrate 60% on data it was not fitted to, using
the lower bound of a 95% confidence interval so a small lucky streak cannot pass.

A gate that has been cleared in the past is evidence, not a guarantee. Markets
change. The gate can be cleared today and the strategy can still lose tomorrow.

## 2. A high win rate can hide a losing strategy

Mean-reversion strategies take many small wins and occasional very large losses.
An 80% win rate paired with losses ten times the size of the wins is a losing
strategy.

The system's own null-market test demonstrates this: on pure noise with no
opportunity in it at all, these rules still "win" 55–60% of the time — while
losing money. Never look at the win rate alone. Profit factor, average profit
per trade, and worst drawdown are all gated for this reason.

## 3. The costs are estimates, and they are optimistic

`arbtool/universe.py` contains commission, spread, tax and borrow assumptions.
They are reasonable defaults, not your broker's numbers. Retail FX conversion
alone is frequently far worse than the 15 basis points assumed here.

**Replace them with your real quoted costs before trusting any result.** A
strategy that is marginal under modelled costs is a losing strategy under real
ones.

## 4. Free daily data has real problems

- Only one price per day. Real execution happens intraday, at prices this
  system never sees.
- Corporate actions — splits, special dividends, ratio changes — can be handled
  inconsistently by free sources, creating fake gaps that look like enormous
  opportunities.
- Delisted companies are missing entirely, so history is biased towards
  survivors.
- The share-ratio check catches the worst of this, and disqualifies pairs whose
  data-implied ratio disagrees with the configured one. It does not catch
  everything.

## 5. You are competing against people who are much better equipped

Cross-listing gaps are watched by firms with co-located servers, direct exchange
feeds, sub-millisecond execution and institutional cost structures. By the time
a gap appears in free end-of-day data, the people who could act on it already
have.

Any edge that survives here is likely to be in slower, smaller, more awkward
corners of the market — which are exactly the ones with the worst liquidity and
the highest real trading costs.

## 6. Shorting is required, and is not always possible

True arbitrage means being long one listing and short the other. Without the
short leg, you are not hedged — you are making a directional bet on a company,
which is a completely different risk.

Short selling requires a margin account, borrowable shares, and the willingness
to accept unlimited theoretical loss. Some venues (Taiwan, for instance) do not
reliably permit it. Borrow can be recalled at the worst moment. Regulators ban
short selling during crises — precisely when spreads are widest and the strategy
looks most attractive.

`--long-only` exists but it changes the strategy into something else, and the
system labels it as unhedged when you use it.

## 7. Gaps can widen instead of closing

The strategy assumes the two prices come back together. Sometimes they do not:

- The dual-listed structure is unified or collapsed (this has happened to real
  companies, more than once).
- Capital controls or tax changes make one line permanently worth more.
- Index inclusion or exclusion permanently shifts demand for one line.
- One listing is suspended while the other keeps trading.

The stop-loss limits the damage. It does not prevent it, and a stop can be
filled far below where it was set.

## 8. The exchanges are not open at the same time

Sydney closes before New York opens. You cannot execute both legs of a BHP trade
simultaneously — you hold one leg naked for hours. That risk is modelled as a
cost, but the cost is an average and the reality is lumpy: overnight news can
move one leg violently while you are unhedged.

## 9. Currency risk

Unless FX hedging is switched on, a profitable spread trade can still lose money
because the exchange rate moved between entry and exit. FX hedging has costs of
its own.

## 10. Overfitting is the default outcome, not the exception

The system tries 144 model configurations per pair. Search enough settings and
something always looks excellent by chance. The defences are: walk-forward
testing, a multiple-testing correction, a null-market test, and parameter
sensitivity checks. They reduce the risk. They do not eliminate it.

If you widen the search grid, you make this worse, not better.

## 11. This tool cannot place real orders — deliberately

`LiveBroker` raises an error by design. Connecting real money would require you
to open a suitable account, implement your broker's API yourself, replace the
modelled costs with real ones, and run in paper mode long enough to have genuine
forward evidence.

I left that step to you on purpose. An automated system that can move real money
should only ever be switched on by someone who understands what it will do when
it is wrong.

## 12. Regulatory and tax obligations are yours

Short selling, cross-border trading, and algorithmic trading are regulated
differently in every jurisdiction, and the rules differ again for retail
accounts. Tax treatment of paired trades, stamp duty, and foreign dividend
withholding are all your responsibility.

## 13. This is not financial advice

This is a research and paper-trading tool. It is not investment advice, it is
not a recommendation to trade anything, and it carries no warranty. Simulated
past performance is even weaker evidence than real past performance, and real
past performance does not predict future results.

Do not risk money you cannot afford to lose entirely.
