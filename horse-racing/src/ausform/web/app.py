"""Local web dashboard.

Deliberately a single self-contained page with no build step and no
external assets, so it runs anywhere Python does. It binds to localhost by
default: this is a personal analysis tool, not a service.
"""

from __future__ import annotations

import datetime as _dt
import pickle
from pathlib import Path
from ..analyse import RaceAnalysis, analyse_race
from ..features import RollingContext

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ausform</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--line:#272b35;--fg:#e6e8ee;--dim:#9aa3b2;
--good:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#58a6ff}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--line:#e1e4e8;
--fg:#1c2128;--dim:#57606a;--good:#1a7f37;--bad:#cf222e;--warn:#9a6700;--accent:#0969da}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px 18px 64px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13px;margin-bottom:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px;margin-bottom:18px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
select,button,input{background:var(--bg);color:var(--fg);border:1px solid var(--line);
border-radius:7px;padding:8px 11px;font-size:14px}
button{cursor:pointer;border-color:var(--accent);color:var(--accent);font-weight:600}
button:hover{background:var(--accent);color:var(--bg)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:640px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap}
th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
.pos{color:var(--good)}.neg{color:var(--bad)}
.note{background:rgba(210,153,34,.12);border-left:3px solid var(--warn);
padding:10px 13px;border-radius:0 7px 7px 0;margin:9px 0;font-size:13px}
.bet{background:rgba(63,185,80,.10);border-left:3px solid var(--good);
padding:10px 13px;border-radius:0 7px 7px 0;margin:8px 0;font-size:14px}
.muted{color:var(--dim);font-size:13px}
.pill{display:inline-block;background:var(--bg);border:1px solid var(--line);
border-radius:20px;padding:2px 10px;font-size:12px;color:var(--dim);margin-right:6px}
ul{margin:6px 0;padding-left:20px}
</style></head><body><div class="wrap">
<h1>ausform</h1>
<div class="sub">Australian thoroughbred race analysis &middot; local instance</div>

<div class="card">
  <div class="row">
    <select id="race"></select>
    <input id="bank" type="number" value="1000" min="10" step="50" title="Bankroll">
    <button onclick="load()">Analyse</button>
  </div>
</div>

<div id="out"></div>

<div class="card">
  <div class="muted">
  <strong>Read this before betting anything.</strong> The market is the best
  single predictor of a horse race that exists. This tool tries to be a small
  correction to it, not a replacement. Most races should return no bet &mdash;
  that is the tool working, not failing. A run of winners proves nothing:
  it takes on the order of 14,000 bets to distinguish a real 5% edge from luck.
  </div>
</div>
</div>
<script>
async function init(){
  const r = await fetch('/api/races').then(r=>r.json());
  const sel = document.getElementById('race');
  sel.innerHTML = r.races.map(x=>`<option value="${x.id}">${x.label}</option>`).join('');
  if(r.races.length) load();
}
function pct(x){return x==null?'&mdash;':(x*100).toFixed(1)+'%'}
function money(x){return x==null?'&mdash;':'$'+x.toFixed(2)}
function signed(x){if(x==null)return '&mdash;';
  const c=x>=0?'pos':'neg';return `<span class="${c}">${(x*100).toFixed(1)}%</span>`}
async function load(){
  const id = document.getElementById('race').value;
  const bank = document.getElementById('bank').value;
  const out = document.getElementById('out');
  out.innerHTML = '<div class="card muted">Analysing&hellip;</div>';
  const d = await fetch(`/api/analyse/${encodeURIComponent(id)}?bankroll=${bank}`)
                  .then(r=>r.json());
  if(d.error){out.innerHTML=`<div class="card note">${d.error}</div>`;return}
  let h = `<div class="card"><h1 style="font-size:17px">${d.title}</h1>
    <div class="sub">${d.subtitle}</div>
    <span class="pill">${d.field_size} runners</span>
    <span class="pill">${d.n_places} places paid</span>
    ${d.overround!=null?`<span class="pill">overround ${(d.overround*100).toFixed(1)}%</span>`:''}
    </div>`;
  h += `<div class="card"><div class="scroll"><table><tr>
    <th>#</th><th>Horse</th><th>Bar</th><th>Model</th><th>Market</th>
    <th>Place</th><th>Fair</th><th>Odds</th><th>Edge</th></tr>`;
  for(const a of d.runners){
    h += `<tr><td>${a.number}</td><td>${a.name}</td><td>${a.barrier??''}</td>
      <td>${pct(a.model_win_prob)}</td><td>${pct(a.market_win_prob)}</td>
      <td>${pct(a.model_place_prob)}</td><td>${money(a.fair_win_odds)}</td>
      <td>${money(a.win_odds)}</td><td>${signed(a.win_edge)}</td></tr>`;
  }
  h += `</table></div></div>`;
  h += `<div class="card"><strong>Recommended bets</strong>`;
  if(d.stakes.length){
    for(const s of d.stakes)
      h += `<div class="bet"><strong>${s.bet_type.toUpperCase()}</strong>
        ${s.selection} &mdash; $${s.amount.toFixed(2)} @ $${s.odds.toFixed(2)}
        <span class="muted">(edge ${(s.edge*100).toFixed(1)}%, ${s.reason})</span></div>`;
  } else { h += `<div class="muted" style="margin-top:8px">No bet.</div>`; }
  for(const w of d.warnings) h += `<div class="note">${w}</div>`;
  h += `</div>`;
  if(d.top_notes && d.top_notes.length){
    h += `<div class="card"><strong>Why ${d.top_name} rates on top</strong>
      <ul>${d.top_notes.map(n=>`<li>${n}</li>`).join('')}</ul></div>`;
  }
  if(d.exotics && d.exotics.length){
    h += `<div class="card"><strong>Exotic combinations</strong>
      <div class="muted" style="margin:6px 0">Probabilities only. Pricing a tote
      exotic needs the pool composition, which is not published before the jump,
      so no expected value is shown &mdash; and the takeout on these is 20&ndash;23%.</div>`;
    for(const e of d.exotics)
      h += `<div class="muted">${e.type}: ${e.selection} &mdash;
        ${(e.probability*100).toFixed(2)}% <em>${e.note}</em></div>`;
    h += `</div>`;
  }
  out.innerHTML = h;
}
init();
</script></body></html>"""


def create_app(db_path: str = "ausform.db", model_path: str = "model.pkl"):
    """Build the FastAPI app.

    Falls back to simulated races when no database is present, so the
    dashboard is usable immediately rather than requiring a data feed you
    may not have yet.
    """
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI is not installed. Run: pip install 'ausform[web]'")

    app = FastAPI(title="ausform", docs_url=None, redoc_url=None)
    state: dict = {"races": [], "model": None, "context": None}

    _load_state(state, db_path, model_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/api/races")
    def races() -> dict:
        return {
            "races": [
                {
                    "id": race.race_id,
                    "label": (f"{race.date} {race.track.name} "
                              f"R{race.race_number} {race.distance_m}m"),
                }
                for race in state["races"][:200]
            ]
        }

    @app.get("/api/analyse/{race_id}")
    def analyse(race_id: str, bankroll: float = 1000.0) -> JSONResponse:
        race = next((r for r in state["races"] if r.race_id == race_id), None)
        if race is None:
            raise HTTPException(status_code=404, detail="Race not found")
        if state["model"] is None:
            return JSONResponse({
                "error": "No trained model available. Run `ausform train` first, "
                         "or `ausform demo` to see the pipeline on simulated data."
            })
        analysis = analyse_race(race, state["model"], state["context"],
                                bankroll=bankroll)
        return JSONResponse(_serialise(analysis))

    return app


def _load_state(state: dict, db_path: str, model_path: str) -> None:
    if Path(model_path).exists():
        payload = pickle.loads(Path(model_path).read_bytes())
        state["model"] = payload.get("model")
        state["context"] = payload.get("context") or RollingContext()
    else:
        state["context"] = RollingContext()

    if Path(db_path).exists():
        from ..data.store import RaceStore
        with RaceStore(db_path) as store:
            state["races"] = sorted(store.iter_races(),
                                    key=lambda r: (r.date, r.race_number),
                                    reverse=True)[:200]

    if not state["races"]:
        from ..data.simulator import SeasonSimulator
        simulator = SeasonSimulator(seed=3, n_horses=4000)
        races = simulator.simulate_season(
            _dt.date.today() - _dt.timedelta(days=200), days=200,
            meetings_per_day=2, races_per_meeting=8)
        state["races"] = races[-60:]

        if state["model"] is None and len(races) > 400:
            state["model"], state["context"] = _quick_train(races[:-60])


def _quick_train(races):
    """Fit a model on the fly so the dashboard is never empty."""
    from ..features import build_features
    from ..model import TwoStageModel, observations_from_featuresets

    context = RollingContext()
    feature_sets = []
    for race in races:
        feature_sets.append(build_features(race, context))
        context.observe(race)

    observations = observations_from_featuresets(feature_sets)
    if len(observations) < 100:
        return None, context

    odds_by_race = {
        r.race_id: [x.fixed_win_odds for x in r.active_runners] for r in races
    }
    model = TwoStageModel.create()
    model.fit(observations, odds_by_race, feature_names=feature_sets[0].names)
    return model, context


def _serialise(analysis: RaceAnalysis) -> dict:
    race = analysis.race
    top = analysis.top_pick
    return {
        "title": (f"{race.track.name} R{race.race_number} "
                  f"{race.distance_m}m"),
        "subtitle": (f"{race.date} &middot; {race.name or 'race'}"
                     + (f" &middot; track {race.track_condition}"
                        if race.track_condition else "")),
        "field_size": race.field_size,
        "n_places": analysis.n_places,
        "overround": analysis.market_overround,
        "runners": [
            {
                "number": a.number,
                "name": a.name,
                "barrier": a.barrier,
                "model_win_prob": a.model_win_prob,
                "market_win_prob": a.market_win_prob,
                "model_place_prob": a.model_place_prob,
                "fair_win_odds": (a.fair_win_odds
                                  if a.fair_win_odds < 1e6 else None),
                "win_odds": a.win_odds,
                "win_edge": a.win_edge,
            }
            for a in analysis.assessments
        ],
        "stakes": [
            {
                "selection": s.selection,
                "bet_type": s.bet_type,
                "amount": s.amount,
                "odds": s.odds,
                "edge": s.edge,
                "reason": s.reason,
            }
            for s in analysis.stakes
        ],
        "warnings": analysis.warnings,
        "exotics": analysis.exotic_suggestions,
        "top_name": top.name if top else "",
        "top_notes": top.notes if top else [],
    }
