"""Registry of Australian racecourses.

Coordinates are used for the weather lookup, so they only need to be
accurate to within a kilometre or so. Track geometry (circumference,
straight length, handedness) feeds the barrier-bias features: a short
straight on a tight left-handed circuit punishes wide gates far more than
a big galloping track like Flemington does.
"""

from __future__ import annotations

from typing import Optional

from .types import Surface, Track

_TRACKS: dict[str, Track] = {}


def _register(track: Track) -> Track:
    _TRACKS[track.code] = track
    return track


# --- Victoria -------------------------------------------------------------
FLEMINGTON = _register(Track("FLEM", "Flemington", "VIC", -37.7887, 144.9126,
                             circumference_m=2312, straight_m=450, direction="left"))
CAULFIELD = _register(Track("CAUL", "Caulfield", "VIC", -37.8812, 145.0416,
                            circumference_m=2080, straight_m=367, direction="left"))
MOONEE_VALLEY = _register(Track("MVAL", "Moonee Valley", "VIC", -37.7656, 144.9287,
                                circumference_m=1805, straight_m=173, direction="left"))
SANDOWN = _register(Track("SAND", "Sandown", "VIC", -37.9089, 145.1712,
                          circumference_m=2200, straight_m=491, direction="left"))
GEELONG = _register(Track("GEEL", "Geelong", "VIC", -38.1685, 144.3542,
                          circumference_m=1970, straight_m=400, direction="left"))
BALLARAT = _register(Track("BLRT", "Ballarat", "VIC", -37.5622, 143.8503,
                           circumference_m=2000, straight_m=400, direction="left"))
BENDIGO = _register(Track("BDGO", "Bendigo", "VIC", -36.757, 144.2794,
                          circumference_m=1800, straight_m=400, direction="left"))
PAKENHAM = _register(Track("PAKM", "Pakenham", "VIC", -38.1567, 145.5711,
                           circumference_m=2000, straight_m=400, direction="left"))
WARRNAMBOOL = _register(Track("WBOL", "Warrnambool", "VIC", -38.3847, 142.4911,
                              circumference_m=1750, straight_m=350, direction="left"))

# --- New South Wales ------------------------------------------------------
RANDWICK = _register(Track("RAND", "Royal Randwick", "NSW", -33.9114, 151.2286,
                           circumference_m=2224, straight_m=410, direction="right"))
ROSEHILL = _register(Track("ROSE", "Rosehill Gardens", "NSW", -33.8242, 151.0225,
                           circumference_m=2048, straight_m=408, direction="right"))
CANTERBURY = _register(Track("CANT", "Canterbury Park", "NSW", -33.9047, 151.1081,
                             circumference_m=1580, straight_m=305, direction="right"))
WARWICK_FARM = _register(Track("WFRM", "Warwick Farm", "NSW", -33.9131, 150.9367,
                               circumference_m=1963, straight_m=330, direction="right"))
NEWCASTLE = _register(Track("NCLE", "Newcastle", "NSW", -32.9283, 151.7817,
                            circumference_m=1900, straight_m=400, direction="right"))
GOSFORD = _register(Track("GOSF", "Gosford", "NSW", -33.4215, 151.3283,
                          circumference_m=1650, straight_m=350, direction="right"))
KEMBLA_GRANGE = _register(Track("KEMB", "Kembla Grange", "NSW", -34.4573, 150.8053,
                                circumference_m=2000, straight_m=400, direction="right"))
HAWKESBURY = _register(Track("HAWK", "Hawkesbury", "NSW", -33.6000, 150.7600,
                             circumference_m=1800, straight_m=400, direction="right"))
WYONG = _register(Track("WYNG", "Wyong", "NSW", -33.2900, 151.4200,
                        circumference_m=1600, straight_m=300, direction="right"))

# --- Queensland -----------------------------------------------------------
EAGLE_FARM = _register(Track("EFRM", "Eagle Farm", "QLD", -27.4338, 153.0678,
                             circumference_m=2048, straight_m=450, direction="right"))
DOOMBEN = _register(Track("DOOM", "Doomben", "QLD", -27.4279, 153.0592,
                          circumference_m=1750, straight_m=340, direction="right"))
GOLD_COAST = _register(Track("GCST", "Gold Coast", "QLD", -28.0089, 153.4042,
                             circumference_m=1900, straight_m=400, direction="right"))
SUNSHINE_COAST = _register(Track("SUNC", "Sunshine Coast", "QLD", -26.7977, 153.061,
                                 circumference_m=1800, straight_m=400, direction="right"))
IPSWICH = _register(Track("IPSW", "Ipswich", "QLD", -27.6200, 152.7600,
                          circumference_m=1700, straight_m=350, direction="right"))
TOOWOOMBA = _register(Track("TOOW", "Toowoomba", "QLD", -27.5700, 151.9500,
                            circumference_m=1700, straight_m=350, direction="right",
                            surface=Surface.SYNTHETIC))

# --- South Australia ------------------------------------------------------
MORPHETTVILLE = _register(Track("MORP", "Morphettville", "SA", -34.9800, 138.5350,
                                circumference_m=2000, straight_m=400, direction="left"))
MURRAY_BRIDGE = _register(Track("MBDG", "Murray Bridge", "SA", -35.1200, 139.2700,
                                circumference_m=1900, straight_m=400, direction="left"))
GAWLER = _register(Track("GAWL", "Gawler", "SA", -34.6000, 138.7400,
                         circumference_m=1800, straight_m=350, direction="left"))

# --- Western Australia ----------------------------------------------------
ASCOT_WA = _register(Track("ASCT", "Ascot", "WA", -31.9370, 115.9250,
                           circumference_m=2000, straight_m=300, direction="left"))
BELMONT = _register(Track("BLMT", "Belmont Park", "WA", -31.9450, 115.9130,
                          circumference_m=1600, straight_m=300, direction="left"))
PINJARRA = _register(Track("PINJ", "Pinjarra", "WA", -32.6300, 115.8700,
                           circumference_m=1800, straight_m=350, direction="left"))

# --- Tasmania / NT / ACT --------------------------------------------------
HOBART = _register(Track("HOBT", "Elwick (Hobart)", "TAS", -42.8300, 147.2900,
                         circumference_m=1700, straight_m=300, direction="left"))
LAUNCESTON = _register(Track("LTON", "Mowbray (Launceston)", "TAS", -41.4400, 147.1400,
                             circumference_m=1600, straight_m=300, direction="left"))
DARWIN = _register(Track("DARW", "Fannie Bay (Darwin)", "NT", -12.4200, 130.8700,
                         circumference_m=1600, straight_m=300, direction="left"))
CANBERRA = _register(Track("CBRA", "Thoroughbred Park", "ACT", -35.2400, 149.1400,
                           circumference_m=1900, straight_m=400, direction="left"))


# Common alternative spellings seen across feeds, normalised to our codes.
_ALIASES: dict[str, str] = {
    "the valley": "MVAL",
    "moonee valley": "MVAL",
    "mv": "MVAL",
    "royal randwick": "RAND",
    "rosehill gardens": "ROSE",
    "canterbury park": "CANT",
    "eagle farm": "EFRM",
    "sandown hillside": "SAND",
    "sandown lakeside": "SAND",
    "elwick": "HOBT",
    "mowbray": "LTON",
    "fannie bay": "DARW",
    "thoroughbred park": "CBRA",
    "belmont park": "BLMT",
    "murray bridge": "MBDG",
    "sunshine coast": "SUNC",
    "gold coast": "GCST",
    "kembla grange": "KEMB",
    "warwick farm": "WFRM",
}


def get_track(identifier: str) -> Optional[Track]:
    """Look up a track by code, full name, or common alias (case-insensitive)."""
    if not identifier:
        return None
    key = identifier.strip()
    if key.upper() in _TRACKS:
        return _TRACKS[key.upper()]
    lowered = key.lower()
    if lowered in _ALIASES:
        return _TRACKS[_ALIASES[lowered]]
    for track in _TRACKS.values():
        if track.name.lower() == lowered:
            return track
    return None


def all_tracks() -> list[Track]:
    return list(_TRACKS.values())


def unknown_track(name: str, state: str = "??") -> Track:
    """Placeholder for a course we have no record of.

    Returned with null-island coordinates so the weather adapter can detect
    it and skip the lookup rather than silently fetching the wrong forecast.
    """
    return Track(code=name.upper()[:4], name=name, state=state,
                 latitude=0.0, longitude=0.0)
