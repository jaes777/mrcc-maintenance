"""Provider interfaces.

Racing data in Australia is fragmented and mostly commercial. Rather than
bind the tool to one vendor, everything is expressed against these two
narrow protocols. Swapping in a paid feed later means writing one class,
not touching the model.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional, Protocol, runtime_checkable

from ..types import Meeting, Race


@runtime_checkable
class FormProvider(Protocol):
    """Supplies race fields and horse form history."""

    def meetings(self, date: _dt.date) -> list[Meeting]:
        """All meetings scheduled on `date`."""
        ...

    def race(self, race_id: str) -> Optional[Race]:
        """A single race, with runners and their form populated."""
        ...


@runtime_checkable
class OddsProvider(Protocol):
    """Supplies market prices for a race."""

    def attach_odds(self, race: Race) -> Race:
        """Populate the odds fields on `race.runners` in place and return it."""
        ...


class ProviderError(RuntimeError):
    """Raised when a provider fails in a way the caller should know about.

    Distinguished from network blips, which providers handle internally by
    returning empty/None so that a single unreachable source degrades the
    analysis rather than aborting it.
    """
