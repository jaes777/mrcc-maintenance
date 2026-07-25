"""Venues and cross-listed instrument pairs.

A *cross-listing* is one company whose shares trade on two different exchanges,
usually in two different currencies. Because both lines are claims on the same
company, their prices must track each other once you convert currency and adjust
for the share ratio. When they drift apart, that gap is the trading opportunity.

The share ratios below are starting points only. Ratios change (companies revise
ADR ratios, do splits, unify dual-listed structures), so the system never trusts
them blindly: :func:`arbtool.pairs.infer_ratio` re-derives the ratio from actual
price history and a pair whose configured ratio disagrees with the data is
disqualified before it can ever be traded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Venue:
    """An exchange, with the frictions that apply when you trade on it."""

    code: str
    name: str
    currency: str
    commission_bps: float          # broker commission, one side, basis points
    half_spread_bps: float         # typical cost of crossing half the bid/ask
    buy_tax_bps: float = 0.0       # e.g. UK stamp duty, HK stamp duty
    sell_tax_bps: float = 0.0
    borrow_bps_per_year: float = 50.0   # cost of borrowing stock to sell short
    utc_open_minutes: int = 0      # session open, minutes past UTC midnight
    utc_close_minutes: int = 0
    shortable: bool = True

    def overlaps(self, other: "Venue") -> bool:
        """Do the two trading sessions overlap at all (UTC)?

        Non-overlapping sessions mean the two legs of a trade cannot be executed
        at the same instant, which is a genuine, unavoidable risk rather than a
        modelling detail.
        """
        return (
            self.utc_open_minutes < other.utc_close_minutes
            and other.utc_open_minutes < self.utc_close_minutes
        )


def _m(hour: int, minute: int = 0) -> int:
    return hour * 60 + minute


VENUES: Dict[str, Venue] = {
    "NYSE": Venue("NYSE", "New York Stock Exchange", "USD", 1.0, 2.0,
                  sell_tax_bps=0.8, utc_open_minutes=_m(14, 30), utc_close_minutes=_m(21)),
    "NASDAQ": Venue("NASDAQ", "Nasdaq", "USD", 1.0, 2.0,
                    sell_tax_bps=0.8, utc_open_minutes=_m(14, 30), utc_close_minutes=_m(21)),
    "LSE": Venue("LSE", "London Stock Exchange", "GBP", 1.5, 3.0,
                 buy_tax_bps=50.0, utc_open_minutes=_m(8), utc_close_minutes=_m(16, 30)),
    "ASX": Venue("ASX", "Australian Securities Exchange", "AUD", 2.0, 4.0,
                 utc_open_minutes=_m(0), utc_close_minutes=_m(6)),
    "XETR": Venue("XETR", "Deutsche Boerse Xetra", "EUR", 1.5, 3.0,
                  utc_open_minutes=_m(7), utc_close_minutes=_m(15, 30)),
    "EURONEXT": Venue("EURONEXT", "Euronext Amsterdam/Paris", "EUR", 1.5, 3.0,
                      utc_open_minutes=_m(7), utc_close_minutes=_m(15, 30)),
    "TSX": Venue("TSX", "Toronto Stock Exchange", "CAD", 2.0, 4.0,
                 utc_open_minutes=_m(14, 30), utc_close_minutes=_m(21)),
    "HKEX": Venue("HKEX", "Hong Kong Exchanges", "HKD", 3.0, 5.0,
                  buy_tax_bps=13.0, sell_tax_bps=13.0,
                  utc_open_minutes=_m(1, 30), utc_close_minutes=_m(8)),
    "TWSE": Venue("TWSE", "Taiwan Stock Exchange", "TWD", 4.0, 6.0,
                  sell_tax_bps=30.0, borrow_bps_per_year=150.0, shortable=False,
                  utc_open_minutes=_m(1), utc_close_minutes=_m(5, 30)),
    "SIX": Venue("SIX", "SIX Swiss Exchange", "CHF", 2.0, 4.0,
                 utc_open_minutes=_m(7), utc_close_minutes=_m(15, 30)),
    "CPH": Venue("CPH", "Nasdaq Copenhagen", "DKK", 2.5, 5.0,
                 utc_open_minutes=_m(7), utc_close_minutes=_m(15)),
    "SYNTH": Venue("SYNTH", "Synthetic test venue", "USD", 1.0, 2.0,
                   utc_open_minutes=_m(0), utc_close_minutes=_m(24 * 60 - 1)),
}


@dataclass(frozen=True)
class Leg:
    """One listing of a company on one venue."""

    symbol: str            # provider-agnostic ticker, e.g. "BHP.AX"
    venue: str
    ordinary_shares: float = 1.0   # ordinary shares represented by one quoted unit
    stooq: Optional[str] = None    # provider-specific symbol overrides
    yahoo: Optional[str] = None

    @property
    def currency(self) -> str:
        return VENUES[self.venue].currency


@dataclass(frozen=True)
class PairSpec:
    """Two listings of the same company that should track each other."""

    pair_id: str
    company: str
    a: Leg
    b: Leg
    kind: str = "adr"      # adr | dual_listed_company | cross_listing
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def venues(self) -> List[str]:
        return [self.a.venue, self.b.venue]

    @property
    def sessions_overlap(self) -> bool:
        return VENUES[self.a.venue].overlaps(VENUES[self.b.venue])

    @property
    def both_shortable(self) -> bool:
        return VENUES[self.a.venue].shortable and VENUES[self.b.venue].shortable

    def describe(self) -> str:
        return (
            f"{self.pair_id}: {self.company} — {self.a.symbol} ({self.a.venue}, "
            f"{self.a.currency}) vs {self.b.symbol} ({self.b.venue}, {self.b.currency})"
        )


def _pair(pair_id, company, a, b, kind="adr", notes="", tags=()) -> PairSpec:
    return PairSpec(pair_id, company, a, b, kind, notes, list(tags))


UNIVERSE: Dict[str, PairSpec] = {
    spec.pair_id: spec
    for spec in [
        _pair("BHP", "BHP Group",
              Leg("BHP", "NYSE", ordinary_shares=2.0, stooq="bhp.us", yahoo="BHP"),
              Leg("BHP.AX", "ASX", stooq="bhp.au", yahoo="BHP.AX"),
              notes="US line is an ADR historically representing 2 ordinary shares.",
              tags=["mining", "no_session_overlap"]),
        _pair("RIO", "Rio Tinto",
              Leg("RIO", "NYSE", stooq="rio.us", yahoo="RIO"),
              Leg("RIO.L", "LSE", stooq="rio.uk", yahoo="RIO.L"),
              kind="dual_listed_company",
              notes="Rio Tinto plc line; a separate Rio Tinto Ltd line trades on ASX.",
              tags=["mining"]),
        _pair("SHEL", "Shell plc",
              Leg("SHEL", "NYSE", ordinary_shares=2.0, stooq="shel.us", yahoo="SHEL"),
              Leg("SHEL.L", "LSE", stooq="shel.uk", yahoo="SHEL.L"),
              tags=["energy"]),
        _pair("BP", "BP plc",
              Leg("BP", "NYSE", ordinary_shares=6.0, stooq="bp.us", yahoo="BP"),
              Leg("BP.L", "LSE", stooq="bp.uk", yahoo="BP.L"),
              tags=["energy"]),
        _pair("AZN", "AstraZeneca",
              Leg("AZN", "NASDAQ", ordinary_shares=0.5, stooq="azn.us", yahoo="AZN"),
              Leg("AZN.L", "LSE", stooq="azn.uk", yahoo="AZN.L"),
              tags=["pharma"]),
        _pair("HSBC", "HSBC Holdings",
              Leg("HSBC", "NYSE", ordinary_shares=5.0, stooq="hsbc.us", yahoo="HSBC"),
              Leg("HSBA.L", "LSE", stooq="hsba.uk", yahoo="HSBA.L"),
              tags=["banks"]),
        _pair("UL", "Unilever",
              Leg("UL", "NYSE", stooq="ul.us", yahoo="UL"),
              Leg("ULVR.L", "LSE", stooq="ulvr.uk", yahoo="ULVR.L"),
              tags=["staples"]),
        _pair("BTI", "British American Tobacco",
              Leg("BTI", "NYSE", stooq="bti.us", yahoo="BTI"),
              Leg("BATS.L", "LSE", stooq="bats.uk", yahoo="BATS.L"),
              tags=["staples"]),
        _pair("SAP", "SAP SE",
              Leg("SAP", "NYSE", stooq="sap.us", yahoo="SAP"),
              Leg("SAP.DE", "XETR", stooq="sap.de", yahoo="SAP.DE"),
              tags=["tech"]),
        _pair("TTE", "TotalEnergies",
              Leg("TTE", "NYSE", stooq="tte.us", yahoo="TTE"),
              Leg("TTE.PA", "EURONEXT", stooq="tte.fr", yahoo="TTE.PA"),
              tags=["energy"]),
        _pair("RY", "Royal Bank of Canada",
              Leg("RY", "NYSE", stooq="ry.us", yahoo="RY"),
              Leg("RY.TO", "TSX", stooq="ry.ca", yahoo="RY.TO"),
              kind="cross_listing", tags=["banks"]),
        _pair("NVS", "Novartis",
              Leg("NVS", "NYSE", stooq="nvs.us", yahoo="NVS"),
              Leg("NOVN.SW", "SIX", stooq="novn.ch", yahoo="NOVN.SW"),
              tags=["pharma"]),
        _pair("NVO", "Novo Nordisk",
              Leg("NVO", "NYSE", stooq="nvo.us", yahoo="NVO"),
              Leg("NOVO-B.CO", "CPH", stooq="novob.dk", yahoo="NOVO-B.CO"),
              tags=["pharma"]),
        _pair("TSM", "TSMC",
              Leg("TSM", "NYSE", ordinary_shares=5.0, stooq="tsm.us", yahoo="TSM"),
              Leg("2330.TW", "TWSE", stooq="2330.tw", yahoo="2330.TW"),
              notes="Taiwan leg is not reliably shortable; long-only mode applies.",
              tags=["tech", "hard_to_short"]),
        _pair("BABA", "Alibaba",
              Leg("BABA", "NYSE", ordinary_shares=8.0, stooq="baba.us", yahoo="BABA"),
              Leg("9988.HK", "HKEX", stooq="9988.hk", yahoo="9988.HK"),
              tags=["tech"]),
    ]
}


def get_pair(pair_id: str) -> PairSpec:
    try:
        return UNIVERSE[pair_id]
    except KeyError:
        raise KeyError(
            f"Unknown pair '{pair_id}'. Known pairs: {', '.join(sorted(UNIVERSE))}"
        ) from None


def selected_pairs(pair_ids: List[str]) -> List[PairSpec]:
    if not pair_ids:
        return list(UNIVERSE.values())
    return [get_pair(pid) for pid in pair_ids]


def synthetic_universe(count: int = 6) -> Dict[str, PairSpec]:
    """Offline stand-in universe used by the synthetic data provider."""
    out: Dict[str, PairSpec] = {}
    for i in range(count):
        pid = f"SYN{i + 1}"
        out[pid] = _pair(
            pid, f"Synthetic Co {i + 1}",
            Leg(f"{pid}.A", "SYNTH"),
            Leg(f"{pid}.B", "SYNTH"),
            kind="cross_listing",
            notes="Generated data for offline testing. Not a real company.",
            tags=["synthetic"],
        )
    return out
