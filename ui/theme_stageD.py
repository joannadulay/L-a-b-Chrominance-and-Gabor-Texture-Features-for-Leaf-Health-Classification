"""
ui/theme_stageD.py — look & feel for the Stage D build.

Deliberately a separate file from the Article 1 baseline's theme.py: both
builds sit in the same project folder, and importing this one only from
the Stage D modules keeps the two designs independent. Nothing here
overwrites or overrides theme.py.

Redesign notes
--------------
The old palette was a generic dashboard navy. This one is built around the
subject matter: a water-and-leaf system living under grow lights. The base
is a deep tank-water charcoal with a green cast, panels sit slightly above
it, and the only saturated colours in the resting state are the leaf-health
accents (green / amber / red). Everything else stays quiet so a result
badge or an out-of-range sensor reading is the single thing that pulls the
eye on an 800x480 touchscreen.

It stays API-compatible with theme.py: every name the Stage D modules
import (COLORS, the threshold constants, make_card,
make_button, status_color) keeps the same signature and behaviour. COLORS
also tolerates keys it doesn't know about instead of raising KeyError, so
older modules that reference a colour that no longer exists still render.
"""

import tkinter as tk

# ---------------------------------------------------------------------------
# Domain thresholds (unchanged — other modules read these)
# ---------------------------------------------------------------------------

TEMP_LOW_C = 22.0
TEMP_HIGH_C = 30.0
PH_LOW = 5.5
PH_HIGH = 7.0
LIGHT_LOW_LUX = 1000.0


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

class _Palette(dict):
    """Dict of colours that degrades gracefully.

    If a module asks for a key this palette doesn't define, guess a sane
    colour from the key name (anything ending in _hover resolves to a
    lighter version of its base) rather than crashing the whole UI over a
    cosmetic lookup.
    """

    def __missing__(self, key):
        if isinstance(key, str) and key.endswith("_hover"):
            base = self.get(key[:-len("_hover")])
            if base:
                value = shade(base, 22)
                self[key] = value
                return value
        if isinstance(key, str) and ("bg" in key or "panel" in key or "card" in key):
            return self["bg_card"]
        return self["text"]


COLORS = _Palette({
    # surfaces — three steps of tank-water charcoal, green-shifted
    "bg": "#0D1411",
    "bg_panel": "#151F1A",
    "bg_card": "#1B2A23",
    "bg_raised": "#22342B",
    "border": "#2C4137",
    "border_soft": "#233329",

    # type
    "text": "#E9F2EC",
    "text_dim": "#B4C6BA",
    "text_muted": "#7F9689",

    # leaf-health accents (the only saturated colours at rest)
    "green": "#57C273",
    "green_hover": "#6BD387",
    "yellow": "#E3A72F",
    "yellow_hover": "#F2B944",
    "orange": "#E07A3F",
    "orange_hover": "#F08C51",
    "red": "#D1495B",
    "red_hover": "#E05C6E",

    # utility / navigation accents
    "teal": "#2FBFA8",
    "teal_hover": "#41D2BA",
    "blue": "#4C8DF6",
    "blue_hover": "#6AA1F8",
    "purple": "#9B6BDB",
    "purple_hover": "#AE82E6",
    "gray": "#3A4A42",
    "gray_hover": "#47584F",

    # semantic aliases
    "accent": "#57C273",
    "ok": "#57C273",
    "warn": "#E3A72F",
    "danger": "#D1495B",
    "disabled": "#3A4A42",
    "disabled_text": "#6C8176",
})


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _clamp(value, low=0, high=255):
    return max(low, min(high, value))


def shade(hex_color: str, amount: int) -> str:
    """Lighten (amount > 0) or darken (amount < 0) a #rrggbb colour."""
    if not isinstance(hex_color, str):
        return "#000000"
    raw = hex_color.lstrip("#")
    if len(raw) != 6:
        return hex_color
    try:
        r, g, b = (int(raw[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return hex_color
    r, g, b = _clamp(r + amount), _clamp(g + amount), _clamp(b + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# backwards-compatible alias (details_window used to define its own _shade)
_shade = shade


def status_color(value, low, high):
    """Colour for a sensor reading against its healthy band.

    Same signature as before: status_color(value, low, high). Returns the
    muted text colour when there's no reading yet, green inside the band,
    amber just outside it, and red well outside it.
    """
    if value is None:
        return COLORS["text_muted"]
    try:
        value = float(value)
        low = float(low)
        high = float(high)
    except (TypeError, ValueError):
        return COLORS["text_muted"]

    if low <= value <= high:
        return COLORS["green"]

    # How far outside the band are we? A near miss reads amber, a big miss
    # reads red, so a reading that's drifting looks different from one
    # that's actually a problem.
    span = high - low
    if not (span > 0) or span == float("inf"):
        span = max(abs(low), 1.0)
    distance = (low - value) if value < low else (value - high)
    return COLORS["yellow"] if distance <= span * 0.25 else COLORS["red"]


# ---------------------------------------------------------------------------
# Widget factories
# ---------------------------------------------------------------------------

def make_card(parent, **kwargs):
    """A panel surface with a hairline border.

    Kept as a plain Frame (same as before) so callers can pack children
    into it directly.
    """
    options = {
        "bg": COLORS["bg_card"],
        "highlightbackground": COLORS["border_soft"],
        "highlightcolor": COLORS["border_soft"],
        "highlightthickness": 1,
        "bd": 0,
    }
    options.update(kwargs)
    return tk.Frame(parent, **options)


def contrast_text(bg: str) -> str:
    """Pick dark or light label text for a given button colour.

    The palette mixes bright accents (green, amber, teal) with muted ones
    (slate grey, red), and a single hard-coded foreground would be
    unreadable on half of them.
    """
    raw = str(bg).lstrip("#")
    if len(raw) != 6:
        return COLORS["text"]
    try:
        r, g, b = (int(raw[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return COLORS["text"]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#0C1310" if luminance > 140 else COLORS["text"]


def make_button(parent, text, bg, hover_bg, command, font=None, state=tk.NORMAL, **kwargs):
    """Flat, touch-sized button.

    Returns a tk.Button so existing code can keep calling
    .config(state=..., bg=...) on it. Two things are handled here so no
    caller has to think about them: the hover colour, and keeping the
    label readable when the caller recolours the button to show a state
    change (an enabled Analyze button and a disabled grey one need
    opposite text colours).
    """
    font = font or ("Helvetica", 10, "bold")

    options = {
        "text": text,
        "command": command,
        "font": font,
        "bg": bg,
        "fg": contrast_text(bg),
        "activebackground": hover_bg,
        "activeforeground": contrast_text(hover_bg),
        "disabledforeground": COLORS["disabled_text"],
        "relief": tk.FLAT,
        "bd": 0,
        "highlightthickness": 0,
        "cursor": "hand2",
        "state": state,
        "padx": 8,
        "pady": 9,
    }
    options.update(kwargs)

    button = tk.Button(parent, **options)
    button._resting_bg = options["bg"]

    original_config = button.config

    def config_with_contrast(*args, **config_kwargs):
        new_bg = config_kwargs.get("bg") or config_kwargs.get("background")
        if new_bg:
            button._resting_bg = new_bg
            config_kwargs.setdefault("fg", contrast_text(new_bg))
            config_kwargs.setdefault("activebackground", shade(new_bg, 18))
        return original_config(*args, **config_kwargs)

    button.config = config_with_contrast
    button.configure = config_with_contrast

    def on_enter(_event):
        if str(button["state"]) != tk.DISABLED:
            button._resting_bg = button.cget("bg")
            original_config(bg=hover_bg, fg=contrast_text(hover_bg))

    def on_leave(_event):
        if str(button["state"]) != tk.DISABLED:
            original_config(bg=button._resting_bg, fg=contrast_text(button._resting_bg))

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

    return button


def make_label(parent, text, size=10, bold=False, fg=None, bg=None, **kwargs):
    """Convenience wrapper so screens don't repeat the same font/colour
    boilerplate on every tk.Label."""
    options = {
        "text": text,
        "font": ("Helvetica", size, "bold" if bold else "normal"),
        "bg": bg or COLORS["bg_card"],
        "fg": fg or COLORS["text"],
        "anchor": "w",
        "justify": "left",
    }
    options.update(kwargs)
    return tk.Label(parent, **options)


def make_divider(parent, color=None, height=1, **kwargs):
    return tk.Frame(parent, bg=color or COLORS["border_soft"], height=height, **kwargs)
