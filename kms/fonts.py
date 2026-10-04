"""Fonts by role, so the graphics look the same everywhere.

On a Mac the original system fonts are used (the graphics were tuned with them).
Everywhere else, or with KMS_FONTS=bundled, the OFL fonts shipped in kms/fonts/ stand in.
"""
import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

BUNDLED = Path(__file__).parent / "fonts"
SYS = "/System/Library/Fonts"
SUP = f"{SYS}/Supplemental"
AVENIR = f"{SYS}/Avenir Next.ttc"
AVENIR_C = f"{SYS}/Avenir Next Condensed.ttc"
HELVETICA = f"{SYS}/HelveticaNeue.ttc"
MENLO = f"{SYS}/Menlo.ttc"

# role -> (macOS system font, collection index), bundled OFL stand-in
ROLES = {
    "impact": ((f"{SUP}/Impact.ttf", 0), "Anton-Regular.ttf"),
    "arial-black": ((f"{SUP}/Arial Black.ttf", 0), "ArchivoBlack-Regular.ttf"),
    "din-condensed-bold": ((f"{SUP}/DIN Condensed Bold.ttf", 0), "BarlowCondensed-Bold.ttf"),
    "din-alternate-bold": ((f"{SUP}/DIN Alternate Bold.ttf", 0), "Barlow-SemiBold.ttf"),
    "avenir-heavy": ((AVENIR, 8), "Barlow-ExtraBold.ttf"),
    "avenir-bold": ((AVENIR, 0), "Barlow-Bold.ttf"),
    "avenir-demi": ((AVENIR, 2), "Barlow-SemiBold.ttf"),
    "avenir-medium": ((AVENIR, 5), "Barlow-Medium.ttf"),
    "avenir-regular": ((AVENIR, 7), "Barlow-Regular.ttf"),
    "avenir-condensed-heavy": ((AVENIR_C, 8), "BarlowCondensed-ExtraBold.ttf"),
    "avenir-condensed-bold": ((AVENIR_C, 0), "BarlowCondensed-Bold.ttf"),
    "helvetica": ((HELVETICA, 0), "Barlow-Regular.ttf"),
    "helvetica-bold": ((HELVETICA, 1), "Barlow-Bold.ttf"),
    "helvetica-medium": ((HELVETICA, 10), "Barlow-Medium.ttf"),
    "menlo": ((MENLO, 0), "SpaceMono-Regular.ttf"),
    "menlo-bold": ((MENLO, 1), "SpaceMono-Bold.ttf"),
    "courier-bold": ((f"{SUP}/Courier New Bold.ttf", 0), "CourierPrime-Bold.ttf"),
    "comic-bold": ((f"{SUP}/Comic Sans MS Bold.ttf", 0), "ComicNeue-Bold.ttf"),
    "rockwell-bold": ((f"{SUP}/Rockwell.ttc", 2), "Arvo-Bold.ttf"),
    "comic-display": ((None, 0), "Bangers-Regular.ttf"),
}


def use_bundled():
    return os.environ.get("KMS_FONTS", "").lower() == "bundled"


@lru_cache(maxsize=None)
def font_path(role):
    """(path, index) for a role: the system font if present, else the bundled stand-in."""
    if role not in ROLES:
        raise KeyError(f"unknown font role {role!r}; known roles: {', '.join(sorted(ROLES))}")
    (system, index), bundled = ROLES[role]
    if system and not use_bundled() and os.path.exists(system):
        return system, index
    return str(BUNDLED / bundled), 0


def font(role, size):
    """A PIL font for a role at a pixel size."""
    path, index = font_path(role)
    return ImageFont.truetype(path, max(1, int(round(size))), index=index)


def fit(role, text, max_width, start=200, min_size=8):
    """The largest font for a role that fits text inside max_width pixels."""
    size = start
    while size > min_size and font(role, size).getlength(text) > max_width:
        size = int(size * max_width / max(font(role, size).getlength(text), 1)) or size - 1
    return font(role, max(size, min_size))
