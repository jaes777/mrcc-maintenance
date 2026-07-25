"""Multi-agent cross-market arbitrage research and paper-trading system.

Researches price gaps between two stock-exchange listings of the same company,
measures whether an edge survives realistic costs and out-of-sample testing,
attacks the result with adversarial agents, and paper-trades what survives.

It does not place real orders. See RISK.md.
"""

__version__ = "1.0.0"

from .config import Config  # noqa: F401
