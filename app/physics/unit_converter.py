from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    si_value: float
    si_unit: str
    raw: str


UNIT_TABLE: dict[str, tuple[str, float]] = {
    "ohm": ("ohm", 1.0),
    "ohms": ("ohm", 1.0),
    "omega": ("ohm", 1.0),
    "Ω": ("ohm", 1.0),
    "ω": ("ohm", 1.0),
    "kohm": ("ohm", 1e3),
    "kilohm": ("ohm", 1e3),
    "kilohms": ("ohm", 1e3),
    "kiloohm": ("ohm", 1e3),
    "kiloohms": ("ohm", 1e3),
    "kilo-ohm": ("ohm", 1e3),
    "kilo-ohms": ("ohm", 1e3),
    "Ω": ("ohm", 1.0),
    "kΩ": ("ohm", 1e3),
    "kΩ": ("ohm", 1e3),
    "kω": ("ohm", 1e3),
    "megohm": ("ohm", 1e6),
    "megohms": ("ohm", 1e6),
    "a": ("A", 1.0),
    "amp": ("A", 1.0),
    "amps": ("A", 1.0),
    "ampere": ("A", 1.0),
    "amperes": ("A", 1.0),
    "ma": ("A", 1e-3),
    "milliamp": ("A", 1e-3),
    "milliamps": ("A", 1e-3),
    "milliampere": ("A", 1e-3),
    "milliamperes": ("A", 1e-3),
    "microamp": ("A", 1e-6),
    "microamps": ("A", 1e-6),
    "microampere": ("A", 1e-6),
    "microamperes": ("A", 1e-6),
    "μa": ("A", 1e-6),
    "ua": ("A", 1e-6),
    "v": ("V", 1.0),
    "volt": ("V", 1.0),
    "volts": ("V", 1.0),
    "mv": ("V", 1e-3),
    "kv": ("V", 1e3),
    "f": ("F", 1.0),
    "farad": ("F", 1.0),
    "farads": ("F", 1.0),
    "mf": ("F", 1e-3),
    "μf": ("F", 1e-6),
    "uf": ("F", 1e-6),
    "microfarad": ("F", 1e-6),
    "microfarads": ("F", 1e-6),
    "nf": ("F", 1e-9),
    "nanofarad": ("F", 1e-9),
    "nanofarads": ("F", 1e-9),
    "pf": ("F", 1e-12),
    "picofarad": ("F", 1e-12),
    "picofarads": ("F", 1e-12),
    "c": ("C", 1.0),
    "coulomb": ("C", 1.0),
    "coulombs": ("C", 1.0),
    "mc": ("C", 1e-3),
    "μc": ("C", 1e-6),
    "uc": ("C", 1e-6),
    "microcoulomb": ("C", 1e-6),
    "microcoulombs": ("C", 1e-6),
    "nc": ("C", 1e-9),
    "nanocoulomb": ("C", 1e-9),
    "nanocoulombs": ("C", 1e-9),
    "j": ("J", 1.0),
    "mj": ("J", 1e-3),
    "μj": ("J", 1e-6),
    "uj": ("J", 1e-6),
    "microjoule": ("J", 1e-6),
    "microjoules": ("J", 1e-6),
    "n": ("N", 1.0),
    "mn": ("N", 1e-3),
    "m": ("m", 1.0),
    "meter": ("m", 1.0),
    "meters": ("m", 1.0),
    "cm": ("m", 1e-2),
    "mm": ("m", 1e-3),
    "millimeter": ("m", 1e-3),
    "millimeters": ("m", 1e-3),
    "w": ("W", 1.0),
    "watt": ("W", 1.0),
    "watts": ("W", 1.0),
    "mw": ("W", 1e-3),
    "kw": ("W", 1e3),
    "hz": ("Hz", 1.0),
    "hertz": ("Hz", 1.0),
    "h": ("H", 1.0),
    "henry": ("H", 1.0),
    "henries": ("H", 1.0),
    "mh": ("H", 1e-3),
    "millihenry": ("H", 1e-3),
    "millihenries": ("H", 1e-3),
    "μh": ("H", 1e-6),
    "uh": ("H", 1e-6),
    "microhenry": ("H", 1e-6),
    "microhenries": ("H", 1e-6),
    "t": ("T", 1.0),
    "tesla": ("T", 1.0),
    "teslas": ("T", 1.0),
    "mt": ("T", 1e-3),
    "millitesla": ("T", 1e-3),
    "milliteslas": ("T", 1e-3),
    "μt": ("T", 1e-6),
    "ut": ("T", 1e-6),
    "microtesla": ("T", 1e-6),
    "microteslas": ("T", 1e-6),
    "wb": ("Wb", 1.0),
    "weber": ("Wb", 1.0),
    "webers": ("Wb", 1.0),
    "mwb": ("Wb", 1e-3),
    "milliweber": ("Wb", 1e-3),
    "milliwebers": ("Wb", 1e-3),
    "μwb": ("Wb", 1e-6),
    "uwb": ("Wb", 1e-6),
    "microweber": ("Wb", 1e-6),
    "microwebers": ("Wb", 1e-6),
    "m²": ("m²", 1.0),
    "m^2": ("m²", 1.0),
    "cm²": ("m²", 1e-4),
    "cm^2": ("m²", 1e-4),
    "mm²": ("m²", 1e-6),
    "mm^2": ("m²", 1e-6),
}


UNIT_PATTERN = "|".join(re.escape(unit) for unit in sorted(UNIT_TABLE, key=len, reverse=True))
NUMBER_PATTERN = r"[-+]?\d*\.?\d+(?:\s*(?:x|×|\*)\s*10\s*(?:\^|\^\{|\{)?\s*[-+]?\d+\}?|(?:e[-+]?\d+)?)?"
NUMBER_WORDS = {
    "one quarter": "0.25",
    "a quarter": "0.25",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


_SUPERSCRIPT_EXP_TRANS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _normalize_superscript_powers_of_ten(text: str) -> str:
    """Normalize Unicode superscript scientific notation.

    Examples:
    - "10⁻⁵" -> "1e-5" (represents 10^(-5))
    - "3×10⁻⁵" -> "3e-5"

    This is intentionally narrow: it only rewrites powers of ten.
    """

    def exp_to_ascii(exp: str) -> str:
        return exp.translate(_SUPERSCRIPT_EXP_TRANS)

    # Coefficient times power-of-ten with superscript exponent.
    text = re.sub(
        r"(?P<coef>\d+(?:\.\d+)?)\s*(?:x|×|\*)\s*10(?P<exp>[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)",
        lambda m: f"{m.group('coef')}e{exp_to_ascii(m.group('exp'))}",
        text,
        flags=re.I,
    )

    # Bare power-of-ten with superscript exponent.
    text = re.sub(
        r"(?<!\d)10(?P<exp>[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)",
        lambda m: f"1e{exp_to_ascii(m.group('exp'))}",
        text,
        flags=re.I,
    )
    return text


def normalize_unit(unit: str) -> str:
    normalized = unit.strip().replace("µ", "μ").replace("Ω", "Ω")
    normalized = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", normalized)
    normalized = normalized.replace(r"\,", "")
    normalized = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", normalized)
    normalized = normalized.replace(r"\Omega", "Ω")
    return normalized.strip()


def parse_number(text: str) -> float:
    compact = text.strip().replace(" ", "").replace("−", "-")
    compact = compact.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+"))
    match = re.fullmatch(r"([-+]?\d*\.?\d+)(?:x|×|\*)10(?:\^|\^\{|\{)?([-+]?\d+)\}?", compact, re.I)
    if match:
        return float(match.group(1)) * (10 ** int(match.group(2)))
    return float(compact)


def normalize_number_words(text: str) -> str:
    normalized = text
    for phrase, value in sorted(NUMBER_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(rf"\b{re.escape(phrase)}\b", value, normalized, flags=re.I)
    return normalized


def convert_value(value: float, unit: str) -> tuple[float, str]:
    normalized = normalize_unit(unit)
    key = normalized.lower()
    if normalized in {"Ω", "kΩ"}:
        key = normalized
    if key not in UNIT_TABLE:
        raise ValueError(f"Unsupported unit: {unit}")
    si_unit, factor = UNIT_TABLE[key]
    return value * factor, si_unit


def parse_quantity(value: str | float, unit: str) -> Quantity:
    numeric = parse_number(str(value)) if isinstance(value, str) else float(value)
    si_value, si_unit = convert_value(numeric, unit)
    return Quantity(value=numeric, unit=normalize_unit(unit), si_value=si_value, si_unit=si_unit, raw=f"{value} {unit}")


def extract_quantities(text: str) -> list[Quantity]:
    # Use a conservative lookahead instead of \b so units like "cm²" match.
    # Avoid f-strings here because regex patterns often need literal '}' characters.
    pattern = re.compile(
        r"(" + NUMBER_PATTERN + r")\s*(" + UNIT_PATTERN + r")(?=$|\s|[\.,;:!\?\)\]\}])",
        re.I,
    )
    quantities: list[Quantity] = []
    normalized_text = normalize_number_words(text).replace("µ", "μ")
    normalized_text = _normalize_superscript_powers_of_ten(normalized_text)
    for match in pattern.finditer(normalized_text):
        try:
            quantities.append(parse_quantity(match.group(1), match.group(2)))
        except ValueError:
            continue
    return quantities


def format_si(value: float, unit: str) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        text = f"{value:.6g}"
    else:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    if unit in {"", "dimensionless"}:
        return text
    return f"{text} {unit}"


SI_TO_SCALED: dict[str, list[tuple[str, float, float, float]]] = {
    "J": [("nJ", 1e9, 0, 1e-6), ("μJ", 1e6, 1e-6, 1e-3), ("mJ", 1e3, 1e-3, 1.0)],
    "F": [("pF", 1e12, 0, 1e-9), ("nF", 1e9, 1e-9, 1e-6), ("μF", 1e6, 1e-6, 1e-3), ("mF", 1e3, 1e-3, 1.0)],
    "C": [("nC", 1e9, 0, 1e-6), ("μC", 1e6, 1e-6, 1e-3), ("mC", 1e3, 1e-3, 1.0)],
    "H": [("μH", 1e6, 0, 1e-3), ("mH", 1e3, 1e-3, 1.0)],
    "V": [("mV", 1e3, 0, 1e-3), ("kV", 1e-3, 1e3, 1e9)],
    "A": [("mA", 1e3, 0, 1e-3)],
    "W": [("mW", 1e3, 0, 1e-3), ("kW", 1e-3, 1e3, 1e9)],
    "N": [("mN", 1e3, 0, 1e-3)],
    "Hz": [("kHz", 1e-3, 1e3, 1e6), ("MHz", 1e-6, 1e6, 1e9)],
    "T": [("μT", 1e6, 0, 1e-3), ("mT", 1e3, 1e-3, 1.0)],
}


def format_best_unit(value: float, si_unit: str) -> str:
    """Convert SI value to most readable scaled unit."""
    if si_unit not in SI_TO_SCALED:
        return format_si(value, si_unit)
    
    abs_val = abs(value)
    for scaled_unit, factor, low, high in SI_TO_SCALED[si_unit]:
        if low <= abs_val < high:
            scaled_value = value * factor
            text = f"{scaled_value:.6g}" if abs(scaled_value) >= 1e4 else f"{scaled_value:.4f}".rstrip("0").rstrip(".")
            return f"{text} {scaled_unit}"
    
    return format_si(value, si_unit)
