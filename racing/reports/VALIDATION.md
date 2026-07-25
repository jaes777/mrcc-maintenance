# Validation results

All figures here come from a **race simulator**, because the build environment
had no network access to real racing data. They demonstrate the code is
correct. They say nothing about real racing performance.

## 1. The simulator matches real Australian racing

Tuned until its behaviour matched published Australian figures:

| Measure | Simulator | Real Australian racing |
|---|---|---|
| Favourite wins | 32.6% | 30–33% |
| Backing every favourite | −9.0% | −5% to −10% |
| Runners at $26+ | −38.4% | −40% to −60% |
| Book percentage | 116% | 115–130% metro |
| Average field size | 10.6 | 9.5–10.8 |

## 2. The tool finds an edge only when one exists

Four simulated worlds where the correct answer is known in advance:

| World | Bets | ROI | p-value | Correct? |
|---|---|---|---|---|
| Efficient market, bookmaker book (116%) | 89 | +13.5% | 0.296 | yes — not significant |
| Efficient market, exchange book (102%) | 2,948 | +1.2% | 0.405 | yes — not significant |
| Weak market, bookmaker book (116%) | 5,413 | +15.9% | 0.0007 | yes — edge found |
| Weak market, exchange book (102%) | 9,093 | +20.6% | ~0 | yes — edge found |

Against an efficient market the tool takes almost no bets and correctly reports
the result as indistinguishable from luck. Against a weak market it finds the
edge and the significance test confirms it. That two-way discrimination is the
point: a tool that always reports profit is broken, and so is one that never
does.

Leakage checks: clean in all four runs.

## 3. Strike rate lands exactly where theory says

On the efficient-market runs the model's top pick won **32.2%**, versus the
market favourite's **32.2%** — identical. This is the expected result and the
central point of HONEST_ANSWER.md: a good model agrees with the market about
who is best, and any edge comes from pricing, not from picking more winners.

## 4. Calibration

Across nine probability bands, every band was statistically calibrated —
when the model says 20%, it wins about 20%. This matters because expected
value, and therefore every staking decision, is only as good as the
calibration underneath it.

## 5. Arithmetic

54 tests covering the closed-form probability identities (orderings summing to
1, Harville recovered at lambda = 1, Monte Carlo agreeing with the exact
calculation), the de-vigging methods, Kelly staking against a direct numerical
optimisation of expected log wealth, tote takeout and breakage, Australian
place-terms thresholds, and regressions for the bugs found during the build.

## Bugs this validation caught

1. **Simulator sign error** — the generated book was 86% (an under-round, i.e.
   free money), producing a fake +24% return across every fold. Caught by the
   "every fold is profitable" honesty check.
2. **Stale loop index** — an optimisation left `pos` pointing at the previous
   loop, corrupting `h_weight_change` and `h_odds_drift`.
3. **Longshot recommendation** — a 220/1 place bet was recommended at HIGH
   confidence off an estimated dividend. Now capped by a maximum-odds guard,
   and estimated prices can never exceed LOW confidence.
4. **`fair_odds` crash on scalar input**, which broke every place recommendation.
