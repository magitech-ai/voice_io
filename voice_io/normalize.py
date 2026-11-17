"""English text front end for spoken-form TTS normalization.

Applies rule-based normalization for numbers, dates, times, money, codes,
identifiers, units, and symbols, with optional NeMo text processing and
lexicon-based respelling for hard names and foreign words.
"""

import logging
import re
from typing import Any

from .respell import Respeller

DIGITS = "zero one two three four five six seven eight nine".split()
ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [
    (10**12, "trillion"),
    (10**9, "billion"),
    (10**6, "million"),
    (1000, "thousand"),
]
MONTHS = (
    "January February March April May June July August September October "
    "November December"
).split()
ORDINAL_IRREGULAR = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

ROMAN_RE = (
    r"(?=[MDCLXVI]+\b)M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
)
ROMAN_CARD = set(
    """chapter chap ch part pt book volume vol act scene section sec phase stage
    level episode season article title class type grade war bowl apollo mark mk
    vatican rocky fantasy wars trek civilization gemini saturn soyuz annex
    appendix figure fig table plate tier round schedule unit division category
    series olympiad psalm canto lesson module form""".split()
)
ROMAN_CARD |= {w + "s" for w in ROMAN_CARD} | {"through", "thru"}
ROMAN_LABEL = set(
    "appendix annex table figure fig plate schedule unit category class type "
    "grade tier section form division series".split()
)
ROMAN_STOP = set(
    "the a an in on of and for to at by from with my our his her their this "
    "that its".split()
)
REGNAL = set(
    """Henry Louis Elizabeth George Edward Richard Charles James William John Paul
    Benedict Pius Leo Gregory Innocent Clement Boniface Urban Alexander Peter
    Ivan Nicholas Frederick Wilhelm Ludwig Philip Felipe Carlos Gustav Gustaf
    Christian Olaf Haakon Harald Ramesses Ramses Thutmose Amenhotep Ptolemy
    Darius Xerxes Cyrus Constantine Justinian Mehmed Suleiman Selim Murad
    Napoleon Umberto Mary Anne Victoria Catherine Isabella Margaret Rama
    Abdullah Robert David Kenneth Duncan Alfonso Ferdinand Sancho Manuel
    Sebastian Stephen Matthias Sigismund Casimir Otto Conrad Rudolf Maximilian
    Leopold Joseph Francis Franz Karl Hugh Baldwin Basil Michael Theodore Edmund
    Harold Magnus Eric Erik Valdemar Frederik Albert Baudouin Philippe Rainier
    Hassan Faisal Fuad Pedro Afonso Jaime Sixtus Julius Adrian Hadrian
    Celestine Honorius Lothair Pepin Antiochus Seleucus Herod Cleopatra
    Wenceslaus Kamehameha Tewodros Menelik Nebuchadnezzar Shalmaneser Sargon
    Tiglath Artaxerxes Andronikos Romanos Leon Mohammed Muhammad
    Charlemagne""".split()
)
TITLES = set(
    "King Queen Pope Emperor Empress Tsar Czar Tsarina Kaiser Pharaoh Sultan "
    "Shah Prince Princess Duke Duchess Count Countess Lord Lady Saint St "
    "Archduke Elector Emir Caliph Khan King's Queen's".split()
)

CURRENCY_CODES = {
    "USD": "$",
    "US$": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "yen",
    "¥": "yen",
    "CNY": "yuan",
    "RMB": "yuan",
    "INR": "rupees",
    "₹": "rupees",
    "Rs.": "rupees",
    "Rs": "rupees",
    "CAD": "Canadian dollars",
    "C$": "Canadian dollars",
    "AUD": "Australian dollars",
    "A$": "Australian dollars",
    "NZD": "New Zealand dollars",
    "CHF": "Swiss francs",
    "HKD": "Hong Kong dollars",
    "HK$": "Hong Kong dollars",
    "SGD": "Singapore dollars",
    "KRW": "won",
    "₩": "won",
    "MXN": "pesos",
    "BRL": "reais",
    "R$": "reais",
    "RUB": "rubles",
    "₽": "rubles",
    "SEK": "Swedish kronor",
    "NOK": "Norwegian kroner",
    "DKK": "Danish kroner",
    "ZAR": "rand",
    "₦": "naira",
    "NGN": "naira",
    "₺": "lira",
    "₱": "pesos",
    "₫": "dong",
    "BTC": "bitcoin",
    "ETH": "ether",
    "₿": "bitcoin",
}
SCALE_WORDS = {
    "k": "thousand",
    "K": "thousand",
    "m": "million",
    "M": "million",
    "mn": "million",
    "mm": "million",
    "bn": "billion",
    "b": "billion",
    "B": "billion",
    "tn": "trillion",
    "T": "trillion",
    "trn": "trillion",
    "thousand": "thousand",
    "million": "million",
    "billion": "billion",
    "trillion": "trillion",
}
FRAC_DENOMINATORS = {2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100}
FRAC_AFTER_RE = (
    r"\s*(?:of\b|an?\b|cups?|c\.|teaspoons?|tsp|tablespoons?|tbsp|inch|inches|"
    r"in\b|in\.|\"|miles?|mi\b|pounds?|lbs?|ounces?|oz|hours?|hrs?|minutes?|"
    r"mins?|seconds?|pints?|quarts?|gallons?|liters?|litres?|kilos?|kg|feet|"
    r"foot|ft|yards?|acres?|sticks?|pieces?|slices?|loaf|tons?|percent|points?|"
    r"shares?|the\b|times\b|marathon|price|full|empty|done|finished|complete)"
)
DATE_BEFORE_RE = (
    r"(?:\b(?:on|by|until|till|from|since|before|after|due|dated?|deadline|"
    r"through|thru|starting|effective|expires?|expiry|born|died|week of|"
    r"shipped|delivered|arrives?|arriving|ordered|paid|placed|scheduled|filed|"
    r"signed|posted|submitted|opens?|closes?|ends?|starts?|begins?|[Mm]onday|"
    r"[Tt]uesday|[Ww]ednesday|[Tt]hursday|[Ff]riday|[Ss]aturday|[Ss]unday|"
    r"Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?[,:]?\s*)$"
)
ABBREVIATIONS = [
    (r"\bDr\.(?=\s+[A-Z])", "Doctor"),
    (r"\bMr\.", "Mister"),
    (r"\bMrs\.", "Missus"),
    (r"\bMs\.", "Miz"),
    (r"\bProf\.(?=\s+[A-Z])", "Professor"),
    (r"\bSt\.(?=\s+[A-Z])", "Saint"),
    (r"(?<=[a-z] )St\.", "Street"),
    (r"\bRd\.", "Road"),
    (r"\bAve\.", "Avenue"),
    (r"\bBlvd\.", "Boulevard"),
    (r"\bApt\.", "Apartment"),
    (r"\b[Aa]pprox\.", "approximately"),
    (r"\betc\.", "et cetera"),
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\bvs\.?(?=\s)", "versus"),
    (r"\b[Nn]o\.(?=\s*\d)", "number"),
    (r"\bdept\.", "department"),
    (r"\bext\.", "extension"),
    (r"(?<=\d)\s?min\.", " minutes"),
    (r"\bmin\.(?=\s+[a-z])", "minimum"),
    (r"\bmin\.", "minutes"),
    (r"\bhr?s?\.(?=\s)", "hours"),
    (r"(?<![\d\s])\s*\b[Ss]ec\.(?=\s*\d)", " section"),
    (r"\b([Ee])st\.(?=\s*\d{4})", r"\1stablished"),
    (r"\bSgt\.", "Sergeant"),
    (r"\bCapt\.", "Captain"),
    (r"\bLt\.", "Lieutenant"),
    (r"\bCol\.", "Colonel"),
    (r"\bGen\.(?=\s+[A-Z])", "General"),
    (r"\bGov\.", "Governor"),
    (r"\bSen\.", "Senator"),
    (r"\bRep\.(?=\s+[A-Z])", "Representative"),
    (r"\bRev\.", "Reverend"),
    (r"\bHon\.", "Honorable"),
    (r"\bJr\.", "Junior"),
    (r"\bSr\.", "Senior"),
    (r"\bLn\.", "Lane"),
    (r"\bDr\.(?=[\s,.]|$)", "Drive"),
    (r"\bCt\.", "Court"),
    (r"\bPl\.", "Place"),
    (r"\bSq\.", "Square"),
    (r"\bHwy\.", "Highway"),
    (r"\bPkwy\.", "Parkway"),
    (r"\bMt\.", "Mount"),
    (r"\bFt\.(?=\s+[A-Z])", "Fort"),
    (r"\bSte\.", "Suite"),
    (r"\bBldg\.", "Building"),
    (r"\bfl\.", "floor"),
    (r"\bFl\.", "Floor"),
    (r"\bRm\.", "Room"),
    (r"\bEst\.", "Estimated"),
    (r"\best\.", "estimated"),
    (r"\bInc\.", "Incorporated"),
    (r"\bLtd\.", "Limited"),
    (r"\bCo\.(?=\s)", "Company"),
    (r"\bCorp\.", "Corporation"),
    (r"\bDept\.", "Department"),
    (r"\bAssn\.", "Association"),
    (r"\bUniv\.", "University"),
    (r"\bqty\.?", "quantity"),
    (r"\bref\.", "reference"),
    (r"\btel\.", "telephone"),
    (r"\bTel\.", "Telephone"),
    (r"\bfig\.", "figure"),
    (r"\bvol\.", "volume"),
    (r"\bpp\.", "pages"),
    (r"\bp\.(?=\s*\d)", "page"),
    (r"\bwk\.", "week"),
    (r"\bmo\.", "month"),
    (r"\byr\.", "year"),
    (r"\bavg\.", "average"),
    (r"\bmax\.", "maximum"),
]
DAYS = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Tues": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Thur": "Thursday",
    "Thurs": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday",
}
MONTH_ABBR = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Sept": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}
SYMBOLS = [
    (r"\s*±\s*", " plus or minus "),
    (r"\s*≥\s*", " greater than or equal to "),
    (r"\s*≤\s*", " less than or equal to "),
    (r"\s*≠\s*", " not equal to "),
    (r"\s*≈\s*", " approximately "),
    (r"\s*÷\s*", " divided by "),
    (r"\s*→\s*", " to "),
    (r"\s*√\s*", " square root of "),
    (r"(?<=\d)\s*%", " percent"),
    (r"\s*‰", " per mille"),
    (r"\s+=\s+", " equals "),
    (r"(?<=\d)\s*<\s*(?=\d)", " less than "),
    (r"(?<=\d)\s*>\s*(?=\d)", " greater than "),
    (r"\s*&\s*", " and "),
]
UNITS = {
    "km": "kilometers",
    "m": "meters",
    "cm": "centimeters",
    "mm": "millimeters",
    "kg": "kilograms",
    "g": "grams",
    "mg": "milligrams",
    "l": "liters",
    "L": "liters",
    "ml": "milliliters",
    "mL": "milliliters",
    "km/h": "kilometers per hour",
    "mph": "miles per hour",
    "GB": "gigabytes",
    "MB": "megabytes",
    "TB": "terabytes",
    "kB": "kilobytes",
    "GHz": "gigahertz",
    "MHz": "megahertz",
    "kW": "kilowatts",
    "kWh": "kilowatt hours",
    "W": "watts",
    "V": "volts",
    "mAh": "milliamp hours",
    "ft": "feet",
    "in": "inches",
    "lb": "pounds",
    "lbs": "pounds",
    "oz": "ounces",
    "h": "hours",
    "min": "minutes",
    "s": "seconds",
    "ms": "milliseconds",
    "mi": "miles",
    "yd": "yards",
    "yds": "yards",
    "Hz": "hertz",
    "kHz": "kilohertz",
    "dB": "decibels",
    "mpg": "miles per gallon",
    "kph": "kilometers per hour",
    "tsp": "teaspoons",
    "tbsp": "tablespoons",
    "gal": "gallons",
    "qt": "quarts",
    "mcg": "micrograms",
    "µg": "micrograms",
    "μg": "micrograms",
    "nm": "nanometers",
    "µm": "micrometers",
    "μm": "micrometers",
    "ns": "nanoseconds",
    "µs": "microseconds",
    "μs": "microseconds",
    "sec": "seconds",
    "secs": "seconds",
    "Mbps": "megabits per second",
    "Gbps": "gigabits per second",
    "kbps": "kilobits per second",
    "kcal": "kilocalories",
    "mmHg": "millimeters of mercury",
    "dL": "deciliters",
    "dl": "deciliters",
    "PB": "petabytes",
    "MW": "megawatts",
    "GW": "gigawatts",
    "GWh": "gigawatt hours",
    "MWh": "megawatt hours",
    "kPa": "kilopascals",
    "hPa": "hectopascals",
    "mmol": "millimoles",
    "fl oz": "fluid ounces",
    "sq ft": "square feet",
    "sq m": "square meters",
    "km2": "square kilometers",
    "m2": "square meters",
    "cc": "C C",
    "mA": "milliamps",
    "ng": "nanograms",
    "pg": "picograms",
    "IU": "I U",
    "mEq": "milliequivalents",
    "kN": "kilonewtons",
    "Nm": "newton meters",
    "TFLOPS": "teraflops",
    "rpm": "R P M",
    "psi": "P S I",
}
SINGULAR_UNITS = {
    "feet": "foot",
    "inches": "inch",
    "hertz": "hertz",
    "kilohertz": "kilohertz",
    "teaspoons": "teaspoon",
    "C C": "C C",
    "R P M": "R P M",
    "P S I": "P S I",
    "teraflops": "teraflop",
    "I U": "I U",
}
MONTH_RE = (
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)"
)
STREET_TYPES = {
    "St": "Street",
    "Rd": "Road",
    "Ave": "Avenue",
    "Av": "Avenue",
    "Blvd": "Boulevard",
    "Ln": "Lane",
    "Dr": "Drive",
    "Ct": "Court",
    "Pl": "Place",
    "Sq": "Square",
    "Hwy": "Highway",
    "Pkwy": "Parkway",
    "Cres": "Crescent",
    "Ter": "Terrace",
    "Terr": "Terrace",
    "Cir": "Circle",
    "Trl": "Trail",
    "Pk": "Park",
}
STREET_RE = (
    r"(Street|Road|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Highway|"
    r"Parkway|Crescent|Terrace|Circle|Way|Close|Row|Walk|Alley|Plaza|Trail|"
    r"Pike|Loop|" + "|".join(STREET_TYPES) + r")\b(\.?)"
)
ID_WORDS = (
    r"[Rr]oom|[Rr]m\.?|[Oo]ffice|[Ss]uite|[Ss]te\.?|[Aa]partment|[Aa]pt\.?|"
    r"[Uu]nit|[Gg]ate|[Pp]latform|[Ff]light|[Bb]us|[Rr]oute|[Hh]ighway|"
    r"[Hh]wy\.?|[Ii]nterstate|[Tt]erminal|[Bb]ay|[Dd]ock|[Pp]ier|[Ss]eat|"
    r"[Cc]abin|[Ww]ard|[Ll]ocker|[Bb]ooth|[Ss]tudio|[Bb]uilding|[Bb]ldg\.?|"
    r"[Ee]xit|[Pp]ort|[Ee]xt\.?|[Ee]xtension|[Mm]odel|[Bb]oeing|[Aa]irbus|"
    r"[Cc]heckpoint|[Hh]angar|[Tt]rain|[Cc]oach|[Ff]erry|[Cc]ar"
)
ELEMENTS = set(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
    Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
    Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr""".split()
)
CODE_CONTEXT_RE = (
    r"(?i:\b(?:codes?|ref(?:erence)?|serial|PIN|password|passcode|quote|plate|"
    r"registration|booking|confirmation|tracking|order|voucher|coupon|promo|"
    r"ticket|ID|key|token|account|policy|claim|case|invoice|SKU|part)\b)"
    r"[^.;!?]*$"
)
SCHEME_SAY = {
    "http": "H T T P",
    "https": "H T T P S",
    "file": "file",
    "git": "git",
    "data": "data",
    "blob": "blob",
    "chrome": "chrome",
    "spotify": "spotify",
    "slack": "slack",
    "zoom": "zoom",
    "steam": "steam",
}
UNIT_PREFIX_RE = r"(?<![\w.])(?<!\w-)(\d(?:[\d,]*\d)?(?:\.\d+)?)"


class NumberVerbalizer:
    """Verbalizes numbers, ordinals, years, fractions, money, times, and dates."""

    @staticmethod
    def digits(text: str) -> str:
        """Reads every digit in ``text`` one by one."""
        return " ".join(DIGITS[int(ch)] for ch in text if ch.isdigit())

    @staticmethod
    def spell(text: str) -> str:
        """Spells out alphanumeric characters as uppercase letters and digits."""
        out: list[str] = []
        for ch in text:
            if ch.isdigit():
                out.append(DIGITS[int(ch)])
            elif ch.isalpha():
                out.append(ch.upper())
        return " ".join(out)

    @classmethod
    def cardinal(cls, value: int) -> str:
        """Converts a non-negative integer to English cardinal words."""
        if value < 20:
            return ONES[value]
        if value < 100:
            rem = value % 10
            return TENS[value // 10] + ("" if rem == 0 else " " + ONES[rem])
        if value < 1000:
            rem = value % 100
            return (
                ONES[value // 100]
                + " hundred"
                + ("" if rem == 0 else " and " + cls.cardinal(rem))
            )
        scale_val, name = 1000, "thousand"
        for cand_val, cand_name in SCALES[:-1]:
            if value >= cand_val:
                scale_val, name = cand_val, cand_name
                break
        rest = value % scale_val
        tail = (
            "" if rest == 0 else (" and " if rest < 100 else " ") + cls.cardinal(rest)
        )
        return cls.cardinal(value // scale_val) + " " + name + tail

    @classmethod
    def ordinal(cls, value: int) -> str:
        """Converts a non-negative integer to English ordinal words."""
        words = cls.cardinal(value).split(" ")
        last = words[-1]
        words[-1] = ORDINAL_IRREGULAR.get(
            last, last[:-1] + "ieth" if last.endswith("y") else last + "th"
        )
        return " ".join(words)

    @classmethod
    def year(cls, value: int) -> str:
        """Reads a 4-digit year naturally (e.g., 2005 -> two thousand five)."""
        if 2000 <= value <= 2009:
            return cls.cardinal(value)
        hi, lo = divmod(value, 100)
        if lo == 0:
            return cls.cardinal(hi) + " hundred"
        if lo < 10:
            return cls.cardinal(hi) + " oh " + ONES[lo]
        return cls.cardinal(hi) + " " + cls.cardinal(lo)

    @classmethod
    def number(cls, text: str) -> str:
        """Verbalizes an integer or decimal string."""
        cleaned = text.replace(",", "")
        if "." in cleaned:
            whole, frac = cleaned.split(".", 1)
            whole_words = cls.cardinal(int(whole)) if whole else "zero"
            frac_words = " ".join(ONES[int(ch)] for ch in frac)
            return f"{whole_words} point {frac_words}"
        return cls.cardinal(int(cleaned))

    @classmethod
    def number_read(cls, text: str) -> str:
        """Reads room/flight/house identifiers naturally (e.g., 402 -> four oh two)."""
        value = int(text)
        if (text[0] == "0" and len(text) > 1) or len(text) > 4:
            return cls.digits(text)
        if value < 100 or (value % 100 == 0 and len(text) == 3) or value % 1000 == 0:
            return cls.cardinal(value)
        hi, lo = divmod(value, 100)
        if lo == 0:
            return cls.cardinal(hi) + " hundred"
        if lo < 10:
            return cls.cardinal(hi) + " oh " + ONES[lo]
        return cls.cardinal(hi) + " " + cls.cardinal(lo)

    @classmethod
    def fraction_words(cls, num: int, den: int, lead: bool = False) -> str:
        """Verbalizes a fraction (e.g., 3/4 -> three quarters)."""
        if den == 2:
            name = "half" if num == 1 else "halves"
        else:
            base = "quarter" if den == 4 else cls.ordinal(den)
            name = base + ("" if num == 1 else "s")
        prefix = "a" if lead and num == 1 else cls.cardinal(num)
        return f"{prefix} {name}"

    def fallback(self, text: str) -> str:
        """Verbalizes remaining numbers, money, times, and ordinals without NeMo."""
        currencies = {
            "$": ("dollar", "dollars", "cent", "cents"),
            "€": ("euro", "euros", "cent", "cents"),
            "£": ("pound", "pounds", "penny", "pence"),
        }

        def money_sub(match: re.Match[str]) -> str:
            one, many, cent_one, cent_many = currencies[match.group(1)]
            whole = int(match.group(2).replace(",", ""))
            frac = match.group(3)
            if frac and len(frac) != 2:
                return f"{self.number(match.group(2) + '.' + frac)} {many}"
            cents = int(frac) if frac else 0
            if whole == 0 and cents:
                return f"{self.cardinal(cents)} {cent_one if cents == 1 else cent_many}"
            out = f"{self.cardinal(whole)} {one if whole == 1 else many}"
            if cents:
                out += f" and {self.cardinal(cents)} {cent_one if cents == 1 else cent_many}"
            return out

        text = re.sub(r"([$€£])(\d[\d,]*)(?:\.(\d+))?\b", money_sub, text)

        def clock_sub(match: re.Match[str]) -> str:
            hour, minute = int(match.group(1)), int(match.group(2))
            meridiem = match.group(3)
            if (
                minute == 0
                and not meridiem
                and re.match(r"\s*(noon|midnight)", match.string[match.end() :])
            ):
                return self.cardinal(hour)
            if minute == 0 and not meridiem and (hour > 12 or hour == 0):
                return ("zero" if hour == 0 else self.cardinal(hour)) + " hundred"
            if minute == 0:
                out = self.cardinal(hour) + ("" if meridiem else " o'clock")
            elif minute < 10:
                out = f"{self.cardinal(hour)} oh {ONES[minute]}"
            else:
                out = f"{self.cardinal(hour)} {self.cardinal(minute)}"
            if meridiem:
                out += " " + meridiem.replace(".", "").upper()
            return out

        text = re.sub(
            r"\b(\d{1,2}):(\d{2})\s*([aApP]\.?[mM])?(?:\.(?=\s+[a-z]))?(?=\W|$)",
            clock_sub,
            text,
        )
        text = re.sub(
            r"\b(\d{1,2})\s*([aApP])\.?[mM](?:\.(?=\s+[a-z]))?(?=\W|$)",
            lambda m: f"{self.cardinal(int(m.group(1)))} {m.group(2).upper()}M",
            text,
        )

        def date_sub(match: re.Match[str]) -> str:
            month, day, yr_str = (
                int(match.group(1)),
                int(match.group(2)),
                match.group(3),
            )
            yr_val = int(yr_str) + (2000 if len(yr_str) == 2 else 0)
            if 12 < month <= 31 and 1 <= day <= 12:
                return (
                    f"the {self.ordinal(month)} of {MONTHS[day - 1]} "
                    f"{self.year(yr_val)}"
                )
            if not (1 <= month <= 12 and 1 <= day <= 31):
                return match.group(0)
            return f"{MONTHS[month - 1]} {self.ordinal(day)} {self.year(yr_val)}"

        text = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", date_sub, text)
        text = re.sub(
            r"\b(" + "|".join(MONTHS) + r")\.? (\d{1,2})\b(?!:)",
            lambda m: f"{m.group(1)} {self.ordinal(int(m.group(2)))}",
            text,
        )
        text = re.sub(r"(\d)\s?°\s?C\b", r"\1 degrees Celsius", text)
        text = re.sub(r"(\d)\s?°\s?F\b", r"\1 degrees Fahrenheit", text)
        text = re.sub(r"(\d)\s?°", r"\1 degrees", text)
        text = re.sub(r"(?<=\w)/(?=\w)", " slash ", text)
        text = re.sub(
            r"\b(\d+)(st|nd|rd|th)\b",
            lambda m: self.ordinal(int(m.group(1))),
            text,
        )
        text = re.sub(
            r"\b(1[1-9]\d\d|20\d\d)\b(?![.,]\d)",
            lambda m: self.year(int(m.group(1))),
            text,
        )
        text = re.sub(r"(?<![\w-])-(?=\d)", "minus ", text)
        text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
        text = re.sub(
            r"(\d+(?:\.\d+)?)\s?%",
            lambda m: self.number(m.group(1)) + " percent",
            text,
        )
        text = re.sub(
            r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b",
            lambda m: self.number(m.group(0)),
            text,
        )
        return (
            text.replace("&", " and ")
            .replace("+", " plus ")
            .replace("=", " equals ")
            .replace("@", " at ")
            .replace("#", " number ")
        )


class Normalizer:
    """Turns raw English text into the spoken-form text the TTS model reads."""

    def __init__(self, use_nemo: bool = True, respell: bool = True) -> None:
        """Initializes the text normalizer.

        Args:
          use_nemo: Whether to try loading NeMo text normalization if installed.
          respell: Whether to apply phonetic respellings from ``respell.tsv``.
        """
        self._verbalizer = NumberVerbalizer()
        self._nemo: Any = None
        self._respeller: Respeller | None = None

        if respell:
            self._respeller = Respeller()
        if use_nemo:
            try:
                logging.disable(logging.WARNING)
                from nemo_text_processing.text_normalization.normalize import (  # type: ignore[import-not-found]
                    Normalizer as NemoNormalizer,
                )

                self._nemo = NemoNormalizer(input_case="cased", lang="en")
            except ImportError:
                pass

    @staticmethod
    def _is_pronounceable(word: str) -> bool:
        """Returns True if an all-caps chunk reads as a word rather than letters."""
        lower = word.lower()
        if len(lower) < 4 or not re.search(r"[aeiouy]", lower):
            return False
        if re.search(r"[^aeiouy]{3,}", lower) and not re.search(
            r"(str|sch|thr|chr|spr|scr|ght|nds|nts)", lower
        ):
            return False
        return bool(
            re.fullmatch(r"(?:[^aeiouy]{0,3}[aeiouy]{1,2})+[^aeiouy]{0,3}", lower)
        )

    @classmethod
    def _code_read(cls, text: str) -> str:
        """Reads dash/slash-separated alphanumeric codes naturally."""
        out: list[str] = []
        for part in re.split(r"[-/]", text):
            if not part:
                continue
            if part.isdigit():
                out.append(
                    part
                    if len(part) <= 2 and part[0] != "0"
                    else NumberVerbalizer.digits(part)
                )
            elif (
                part.isalpha()
                and part.islower()
                and len(part) > 3
                and cls._is_pronounceable(part)
            ):
                out.append(part)
            elif part.isalpha() and cls._is_pronounceable(part):
                out.append(part.capitalize())
            else:
                out.append(NumberVerbalizer.spell(part))
        return " ".join(out)

    @staticmethod
    def _split_identifier(text: str) -> str:
        """Splits camelCase/snake_case identifiers into spoken words."""
        spaced = re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            " ",
            text.replace("_", " "),
        )
        return " ".join(
            NumberVerbalizer.spell(word) if word.isupper() and len(word) <= 4 else word
            for word in spaced.split()
        )

    @classmethod
    def _normalize_handle(cls, handle: str) -> str:
        """Formats social handles such as ``@Jane_Doe``."""
        out: list[str] = []
        for part in handle.split("_"):
            match = re.fullmatch(r"([a-z]{3,}[nlyex])(ai|hq)", part)
            if match:
                out.append(
                    match.group(1) + " " + NumberVerbalizer.spell(match.group(2))
                )
            else:
                out.append(cls._split_identifier(part))
        return " underscore ".join(item for item in out if item)

    @staticmethod
    def _roman_value(text: str) -> int:
        """Converts a Roman numeral string to an integer."""
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        return sum(
            -values[cur] if nxt and values[nxt] > values[cur] else values[cur]
            for cur, nxt in zip(text, list(text[1:]) + [None])
        )

    @classmethod
    def _roman_sub(cls, match: re.Match[str]) -> str:
        """Replaces contextual Roman numerals with cardinals or regnal ordinals."""
        prev, num = match.group(1), match.group(3)
        if not num:
            return match.group(0)
        before = match.string[: match.start()].split()
        prev2 = (
            before[-1] if before and not re.search(r"[.,;:!?)\"]$", before[-1]) else ""
        )
        key = prev.lower().rstrip(".")
        value = cls._roman_value(num)

        def is_title(word: str) -> bool:
            return bool(re.fullmatch(r"[A-Z][a-z]+", word))

        if key in ("through", "thru", "to", "and", "or") and not re.search(
            r"\b(?:" + "|".join(ROMAN_CARD) + r")\s+\S+\s+$",
            match.string[: match.start()],
            re.I,
        ):
            return match.group(0)
        if key in ROMAN_CARD and (
            len(num) > 1
            or (num == "I" and prev[:1].isupper())
            or (num in "VX" and key not in ROMAN_LABEL)
        ):
            return f"{prev}{match.group(2)}{NumberVerbalizer.cardinal(value)}"
        if not re.fullmatch(r"[IVXL]+", num):
            return match.group(0)
        if prev in REGNAL or (prev2 in TITLES and is_title(prev) and num != "I"):
            return f"{prev}{match.group(2)}the {NumberVerbalizer.ordinal(value)}"
        if (
            len(num) > 1
            and is_title(prev)
            and not re.fullmatch(r"[IVXL]*L[IVXL]*", num)
        ):
            full = is_title(prev2) and prev2.lower() not in ROMAN_STOP
            suffix = (
                f"the {NumberVerbalizer.ordinal(value)}"
                if full
                else NumberVerbalizer.cardinal(value)
            )
            return f"{prev}{match.group(2)}{suffix}"
        return match.group(0)

    @staticmethod
    def _singularize_unit(word: str) -> str:
        """Singularizes a unit phrase before ``per``."""
        head, sep, rest = word.partition(" per ")
        if " of " in head:
            return word
        words = head.split(" ")
        last = words[-1]
        words[-1] = SINGULAR_UNITS.get(last, last[:-1] if last.endswith("s") else last)
        return " ".join(words) + sep + rest

    @classmethod
    def _format_unit(cls, key: str, count_str: str) -> str:
        """Formats a unit name for singular or plural count."""
        return cls._singularize_unit(UNITS[key]) if count_str == "1" else UNITS[key]

    def _normalize_tokens(self, text: str) -> str:
        """Normalizes symbols, money ranges, decades, Roman numerals, and codes."""
        text = (
            text.replace("−", "-")
            .replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
        )
        text = re.sub(r"(?<=\d)([½¼¾⅓⅔⅛])", r" \1", text)
        frac_map = {
            "½": "1/2",
            "¼": "1/4",
            "¾": "3/4",
            "⅓": "1/3",
            "⅔": "2/3",
            "⅛": "1/8",
        }
        text = re.sub(r"[½¼¾⅓⅔⅛]", lambda m: frac_map[m.group(0)], text)
        sup_table = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
        text = re.sub(
            r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "^" + m.group(0).translate(sup_table), text
        )

        text = re.sub(
            r"(?<![\w.])([A-Za-z]*)\.NET\b",
            lambda m: (
                (NumberVerbalizer.spell(m.group(1)) + " " if m.group(1) else "")
                + "dot net"
            ),
            text,
        )
        text = re.sub(r"\b([A-Za-z])#(?!\w)", r"\1 sharp", text)
        text = re.sub(r"\b([A-Za-z])\+\+(?!\w)", r"\1 plus plus", text)
        text = re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)",
            lambda m: self._split_identifier(m.group(1)),
            text,
        )
        text = re.sub(
            r"\b[a-z]{2,}(?:[A-Z][a-z]+){2,}\b",
            lambda m: self._split_identifier(m.group(0)),
            text,
        )
        text = re.sub(
            r"\bv(\d+(?:\.\d+)+)\b",
            lambda m: "version " + " point ".join(m.group(1).split(".")),
            text,
        )

        text = re.sub(
            r"(?<![\w&#])#([A-Za-z][A-Za-z_]*)\b(?![\d-])",
            lambda m: "hashtag " + self._split_identifier(m.group(1)),
            text,
        )
        text = re.sub(
            r"(?<![\w.@])@([A-Za-z_][A-Za-z0-9_]{1,30})\b",
            lambda m: "at " + self._normalize_handle(m.group(1)),
            text,
        )

        text = re.sub(
            r"\b([A-Z]{1,2}\d[A-Z\d]?|[A-Z]\d[A-Z])\s(\d[A-Z]{2}|\d[A-Z]\d)\b",
            lambda m: (
                NumberVerbalizer.spell(m.group(1))
                + ", "
                + NumberVerbalizer.spell(m.group(2))
            ),
            text,
        )
        text = re.sub(
            r"\b((?:ending|ends) in|last (?:four|4)(?: digits)?(?: of)?|PIN|"
            r"[Pp]in|passcode|OTP|[Cc]ode)(\s+)(\d{3,8}(?:\s\d{2,4})*)\b"
            r"(?![.,]\d|-\w)",
            lambda m: (
                m.group(1)
                + m.group(2)
                + ", ".join(NumberVerbalizer.digits(g) for g in m.group(3).split())
            ),
            text,
        )

        text = re.sub(r"\b([A-Z]):\\", r"\1 colon backslash ", text)
        text = re.sub(r"\\", " backslash ", text)
        text = re.sub(r"(?<=\s)\*(?=\s)(?!\s*\d)", "star", text)
        text = re.sub(
            r"(?<![\w.])\.(env|gitignore|bashrc|zshrc|npmrc|htaccess|dockerignore)\b",
            r"dot \1",
            text,
        )
        text = re.sub(
            r"\b(?=\w*_)(?![\w]*\.\w)[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b",
            lambda m: " underscore ".join(
                (
                    NumberVerbalizer.spell(p)
                    if re.search(r"\d", p) and re.search(r"[A-Za-z]", p)
                    else p
                )
                for p in m.group(0).split("_")
            ),
            text,
        )

        text = re.sub(
            r"(?<![\w+])1[-. ](\d{3})[-. ]"
            r"(?:(\d{3})[-. ](\d{4})|([A-Z]{3,}(?:-[A-Z]+)*))\b",
            lambda m: (
                "one "
                + (
                    NumberVerbalizer.cardinal(int(m.group(1)))
                    if m.group(1) in ("800", "900")
                    else NumberVerbalizer.digits(m.group(1))
                )
                + ", "
                + (
                    NumberVerbalizer.digits(m.group(2))
                    + ", "
                    + NumberVerbalizer.digits(m.group(3))
                    if m.group(2)
                    else " ".join(w.capitalize() for w in m.group(4).split("-"))
                )
            ),
            text,
        )

        num_re = r"(\d[\d,]*(?:\.\d+)?)"
        scale_re = (
            r"(\s?(?:thousand|million|billion|trillion)\b|"
            r"(?:k|K|mn|m|M|bn|b|B|tn|T)\b)?"
        )
        codes = "|".join(
            re.escape(k) for k in sorted(CURRENCY_CODES, key=len, reverse=True)
        )

        def format_currency(code: str, num: str, scale: str | None) -> str:
            name = CURRENCY_CODES[code]
            if len(name) == 1:
                return name + num + (scale or "")
            whole, _, cents = num.partition(".")
            if not scale and len(cents) == 2 and name.endswith("dollars"):
                tail = f" and {int(cents)} cents" if int(cents) else ""
                return f"{whole} {name}{tail}"
            scale_word = f" {SCALE_WORDS[scale.strip()]}" if scale else ""
            return f"{num}{scale_word} {name}"

        text = re.sub(
            r"(?<![\w$])(" + codes + r")\s?" + num_re + scale_re,
            lambda m: format_currency(m.group(1), m.group(2), m.group(3)),
            text,
        )
        text = re.sub(
            num_re
            + scale_re
            + r"\s?(USD|EUR|GBP|JPY|CNY|INR|CAD|AUD|CHF|NZD|HKD|SGD|KRW|MXN|BRL|"
            r"RUB|SEK|NOK|DKK|ZAR|BTC|ETH)\b",
            lambda m: format_currency(m.group(3), m.group(1), m.group(2)),
            text,
        )
        text = re.sub(r"(?<!\w)(\d+)\s?¢", r"\1 cents", text)
        currency_words = {"$": "dollars", "€": "euros", "£": "pounds"}

        def money_range(match: re.Match[str]) -> str:
            sym, a, s1, b, s2 = match.groups()
            if not (s1 or s2) and re.search(r"\.\d\d\b", a + " " + b):
                return f"{sym}{a} to {sym}{b}"
            s1_str = " " + SCALE_WORDS[s1.strip()] if s1 else ""
            s2_str = " " + SCALE_WORDS[s2.strip()] if s2 else ""
            left_scale = s1_str if s1_str != s2_str else ""
            return f"{a}{left_scale} to {b}{s2_str} {currency_words[sym]}"

        text = re.sub(
            r"([$€£])" + num_re + scale_re + r"\s?[-–]\s?[$€£]?" + num_re + scale_re,
            money_range,
            text,
        )
        text = re.sub(
            r"([$€£])" + num_re + r"(\s?(?:thousand|million|billion|trillion)\b|"
            r"(?:k|K|mn|m|M|bn|b|B|tn|T)\b)",
            lambda m: (
                f"{m.group(2)} {SCALE_WORDS[m.group(3).strip()]} "
                f"{currency_words[m.group(1)]}"
            ),
            text,
        )

        def plural(num_str: str, unit_word: str) -> str:
            return f"{num_str} {unit_word}{'' if num_str == '1' else 's'}"

        text = re.sub(
            r"\b(\d+)\s?h(?:rs?)?\s?(\d{1,2})(?:\s?m(?:ins?)?)?\b",
            lambda m: plural(m.group(1), "hour") + " " + plural(m.group(2), "minute"),
            text,
        )
        text = re.sub(
            r"\b(\d+)\s?m(?:ins?)?\s?(\d{1,2})\s?s(?:ecs?)?\b",
            lambda m: plural(m.group(1), "minute") + " " + plural(m.group(2), "second"),
            text,
        )
        text = re.sub(
            r"(\b(?:for|last|lasts|lasted|lasting|takes?|took|taking|every|within|"
            r"wait|waited|remaining|runtime|ETA|blocked)\s+(?:\w+\s+){0,2}?)"
            r"(\d{1,3})m\b(?!\s*(?:dash|sprint|race|swim|hurdles|pool|freestyle|"
            r"relay))",
            lambda m: m.group(1) + plural(m.group(2), "minute"),
            text,
        )

        def decade_sub(match: re.Match[str]) -> str:
            digits_str, val = match.group(2), int(match.group(2))
            if val % 10:
                return match.group(0)
            if len(digits_str) == 4:
                word = NumberVerbalizer.year(val)
            elif val == 0 and match.group(1):
                word = "two thousand"
            elif val:
                word = NumberVerbalizer.cardinal(val)
            else:
                word = ""
            if not word:
                return match.group(0)
            return word[:-1] + "ies" if word.endswith("y") else word + "s"

        text = re.sub(
            r"(?<![\w$€£.,'])(')?(\d{2}|1\d{3}|20\d{2})'?s\b", decade_sub, text
        )
        text = re.sub(
            r"(?:(?<=\s)|^)'(\d{2})\b(?!['\w])",
            lambda m: (
                ("oh " + ONES[int(m.group(1))]) if m.group(1)[0] == "0" else m.group(1)
            ),
            text,
        )

        text = re.sub(
            r"\bWW\s?(II|I|2|1)\b",
            lambda m: "World War " + ("Two" if m.group(1) in ("II", "2") else "One"),
            text,
        )
        text = re.sub(
            r"\b([A-Za-z][A-Za-z']*\.?)(\s+)(" + ROMAN_RE + r")\b(?!'\w|-\d)",
            self._roman_sub,
            text,
        )
        text = re.sub(
            r"(?<=[a-z])\((I|II|III|IV|V|VI)\)",
            lambda m: " " + NumberVerbalizer.cardinal(self._roman_value(m.group(1))),
            text,
        )

        text = re.sub(r"(\d)\s?[x×*]\s?(?=10\s?\^)", r"\1 times ", text)
        text = re.sub(
            UNIT_PREFIX_RE + r"\s?(km|m|cm|mm|ft|mi|yd|in)\^([23])\b",
            lambda m: (
                f"{m.group(1)} "
                f"{'square' if m.group(3) == '2' else 'cubic'} "
                f"{UNITS[m.group(2)]}"
            ),
            text,
        )

        def power_sub(match: re.Match[str]) -> str:
            base, exp = match.group(1), match.group(2)
            neg, exp = exp.startswith("-"), exp.lstrip("+-")
            base = " ".join(base) if base.isalpha() and len(base) > 1 else base
            if not exp.isdigit():
                return f"{base} to the {exp}"
            val = int(exp)
            if val == 2 and not neg:
                return f"{base} squared"
            if val == 3 and not neg:
                return f"{base} cubed"
            return (
                f"{base} to the {'minus ' if neg else ''}"
                f"{NumberVerbalizer.ordinal(val)}"
            )

        text = re.sub(
            r"(\b[A-Za-z]{1,3}|\b\d+(?:\.\d+)?|\))\s?\^\s?\(?([+-]?(?:\d+|[a-z]))\)?",
            power_sub,
            text,
        )
        if re.search(r"[=^+]|\bsolve\b|\bequation", text):
            text = re.sub(r"(?<=[\w)])\s+[-–]\s+(?=[\w(])", " minus ", text)
            text = re.sub(r"\b(\d+)([a-fi-jn-rt-z])\b", r"\1 \2", text)
        text = re.sub(
            r"\b(\d+(?:\.\d+)?)x(?=\s+(?:faster|slower|more|less|larger|smaller|"
            r"bigger|higher|lower|cheaper|better|worse|longer|shorter|the|as|over|"
            r"speed|zoom)\b)",
            r"\1 times",
            text,
        )
        text = re.sub(r"(\d)\s+[x×*]\s+(\d+(?:\.\d+)?)(?=\s*=)", r"\1 times \2", text)
        text = re.sub(
            r"(?<![\w$€£.])(\d+\.\d+|\d+(?=\s?\w+\s+[a-z]{3,}s\b))"
            r"\s?(M|bn|mn|tn)\b(?=\s+[a-z]{3,})",
            lambda m: f"{m.group(1)} {SCALE_WORDS[m.group(2)]}",
            text,
        )
        return text

    def _normalize_pre(self, text: str) -> str:
        """Runs pre-NeMo normalization passes."""
        text = self._normalize_tokens(text)
        text = re.sub(
            r"\b(localhost|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}):(\d{2,5})\b",
            lambda m: f"{m.group(1)} colon {NumberVerbalizer.number_read(m.group(2))}",
            text,
        )

        def say_scheme(scheme: str) -> str:
            lower = scheme.lower()
            if lower in SCHEME_SAY:
                return SCHEME_SAY[lower]
            return (
                lower
                if self._is_pronounceable(scheme)
                else NumberVerbalizer.spell(scheme)
            )

        text = re.sub(
            r"(?<![\w.+-])([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\s,.;:!?)\"']|$)",
            lambda m: say_scheme(m.group(1)) + " colon slash slash",
            text,
        )
        text = re.sub(
            r"(?<![\w.+-])(?!https?://)([A-Za-z][A-Za-z0-9+.-]{1,9})://"
            r"(?=[\w/@\[])((?![\w.-]*@)[A-Za-z][\w-]*(?:\.[\w-]+)+\b)?",
            lambda m: (
                say_scheme(m.group(1))
                + " colon slash slash "
                + (m.group(2) or "").replace(".", " dot ").replace("-", " dash ")
            ),
            text,
        )
        text = re.sub(r"\bmailto:(?=[\w.+-]+@)", "mail to ", text)
        text = re.sub(
            r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b",
            lambda m: (
                m.group(1)
                .replace(".", " dot ")
                .replace("_", " underscore ")
                .replace("-", " dash ")
                + " at "
                + m.group(2).replace(".", " dot ").replace("-", " dash ")
            ),
            text,
        )
        tld_re = (
            r"\.(?:com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|us|eu|de|fr|"
            r"info|biz)$"
        )
        text = re.sub(
            r"\bhttps?://(?![\w.-]*[:@])([A-Za-z][\w-]*(?:\.[\w-]+)+\b)?",
            lambda m: (
                re.sub(
                    r"^www ",
                    "w w w ",
                    (m.group(1) or "").replace(".", " dot ").replace("-", " dash "),
                )
                if m.group(1) and not re.search(tld_re, m.group(1))
                else m.group(1) or ""
            ),
            text,
        )
        text = re.sub(r"\bwww\.", "w w w dot ", text)
        text = re.sub(
            r"\b((?:[\w-]+\.)+)(com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|"
            r"us|eu|de|fr|info|biz)\b",
            lambda m: (
                m.group(1).replace(".", " dot ").replace("-", " dash ") + m.group(2)
            ),
            text,
        )

        text = re.sub(
            r"\b([\w-]+)\.(pdf|png|jpe?g|gif|docx?|xlsx?|csv|txt|zip|mp3|mp4|wav|"
            r"json|py|js|html?|gradle|ipynb|rtf|pptx?|webm|bin|dat|bat|ps1|vue|"
            r"jsx|kts|cs|dart|lua|r)\b",
            lambda m: re.sub(r"_", " underscore ", m.group(1)) + " dot " + m.group(2),
            text,
        )

        def fname_sub(match: re.Match[str]) -> str:
            base, ext = match.group(1), match.group(2)
            base = (
                base.replace("_", " underscore ")
                .replace("/", " slash ")
                .replace("-", " dash ")
            )
            base = re.sub(r"\.", " dot ", base)
            return base + " dot " + ext

        text = re.sub(
            r"(?<![\w@])((?:/?[\w-]+/)*[\w-]*[a-z_][\w-]*)\.(conf|log|cfg|ini|yaml|"
            r"yml|toml|md|sh|ts|tsx|css|xml|sql|db|bak|tmp|exe|dmg|apk|iso|tar|gz|"
            r"rar|7z|svg|webp|heic|mov|avi|mkv|flac|ogg|m4a|key|pem|crt|env|lock|"
            r"rs|go|java|kt|swift|rb|php|c|cpp|h|git)\b",
            fname_sub,
            text,
        )
        text = re.sub(
            r"(?<![\w])/((?:[\w.-]+/)*[\w.-]+)",
            lambda m: " slash " + m.group(1).replace("/", " slash "),
            text,
        )

        text = re.sub(r"\b(\d{1,2})-(\d{1,2})-((?:19|20)\d\d)\b", r"\1/\2/\3", text)
        text = re.sub(
            r"\b(\d{1,2})\.(\d{1,2})\.((?:19|20)\d\d)\b",
            lambda m: (
                f"the {NumberVerbalizer.ordinal(int(m.group(1)))} of "
                f"{MONTHS[int(m.group(2)) - 1]} {m.group(3)}"
                if 1 <= int(m.group(1)) <= 31 and 1 <= int(m.group(2)) <= 12
                else m.group(0)
            ),
            text,
        )

        text = re.sub(
            r"\b(" + "|".join(DAYS) + r")\.?\s*[-–]\s*(" + "|".join(DAYS) + r")\.?",
            lambda m: DAYS[m.group(1)] + " to " + DAYS[m.group(2)],
            text,
        )
        text = re.sub(
            r"(?<!\bthe )(?<!\bThe )\b(" + "|".join(DAYS) + r")\.(?=[\s,]|$)",
            lambda m: (
                DAYS[m.group(1)]
                + ("." if re.match(r"\s*(?:$|[A-Z])", m.string[m.end() :]) else "")
            ),
            text,
        )
        text = re.sub(
            r"\b(on|by|until|till|every|from|next|this|last) ("
            + "|".join(DAYS)
            + r")\b(?!')",
            lambda m: m.group(1) + " " + DAYS[m.group(2)],
            text,
        )
        text = re.sub(
            r"\b(" + "|".join(MONTH_ABBR) + r")\.(?=\s*\d)",
            lambda m: MONTH_ABBR[m.group(1)],
            text,
        )
        text = re.sub(
            r"\b(" + "|".join(MONTH_ABBR) + r")\b(?=\s+\d{1,4}\b(?!:))",
            lambda m: MONTH_ABBR[m.group(1)],
            text,
        )
        text = re.sub(
            r"(?<=\d )(" + "|".join(MONTH_ABBR) + r")\b\.?",
            lambda m: MONTH_ABBR[m.group(1)],
            text,
        )

        def street_sub(match: re.Match[str]) -> str:
            num, letter, name, kind, dot = match.groups()
            tail = (
                "."
                if dot and re.match(r"\s*(?:$|[A-Z\n])", match.string[match.end() :])
                else ""
            )
            letter_str = " " + letter.upper() if letter else ""
            return (
                f"{NumberVerbalizer.number_read(num)}{letter_str} {name} "
                f"{STREET_TYPES.get(kind, kind)}{tail}"
            )

        text = re.sub(
            r"\b(\d{1,4})([A-Za-z]?)\s+((?:(?:North|South|East|West|N\.?|S\.?|"
            r"E\.?|W\.?)\s+)?(?:[A-Z][\w']*|\d+(?:st|nd|rd|th))(?:\s+[A-Z][\w']*){0,2})"
            r"\s+" + STREET_RE,
            street_sub,
            text,
        )
        text = re.sub(
            r"(?<!Mount )(?<!Port )(?<!Fort )(?<!Lake )(?<!Cape )"
            r"\b([A-Z][a-z]+|\d+(?:st|nd|rd|th))\s+"
            r"(St|Rd|Ave|Blvd|Ln|Hwy|Pkwy|Cres|Terr?)\b(\.?)",
            lambda m: (
                m.group(0)
                if m.group(1).lower() in ROMAN_STOP | {"on", "at", "visit", "see"}
                or (m.group(2) == "St" and re.match(r"\s+[A-Z]", m.string[m.end() :]))
                else f"{m.group(1)} {STREET_TYPES[m.group(2)]}"
                + (
                    "."
                    if m.group(3) and re.match(r"\s*(?:$|\n)", m.string[m.end() :])
                    else ""
                )
            ),
            text,
        )
        dir_map = {"N": "North", "S": "South", "E": "East", "W": "West"}
        text = re.sub(
            r"\b((?:I|US|SR|Route|Highway|Hwy|Interstate)[- ]\d+)\s([NSEW])\b",
            lambda m: m.group(1) + " " + dir_map[m.group(2)],
            text,
        )
        for pattern, replacement in ABBREVIATIONS:
            text = re.sub(pattern, replacement, text)

        text = re.sub(
            r"\b(\d{4})-(\d{2})-(\d{2})\b",
            lambda m: (
                f"{MONTHS[int(m.group(2)) - 1]} {int(m.group(3))}, {m.group(1)}"
                if 1 <= int(m.group(2)) <= 12
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s*[-–]\s*(\d{1,2})(?:st|nd|rd|th)?\s+"
            + MONTH_RE
            + r"\b",
            lambda m: (
                f"the {NumberVerbalizer.ordinal(int(m.group(1)))} to the "
                f"{NumberVerbalizer.ordinal(int(m.group(2)))} of {m.group(3)}"
                if int(m.group(1)) < int(m.group(2)) <= 31
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MONTH_RE + r"\b",
            lambda m: (
                ("" if re.search(r"\bthe\s+$", m.string[: m.start()], re.I) else "the ")
                + f"{NumberVerbalizer.ordinal(int(m.group(1)))} of {m.group(2)}"
                if 1 <= int(m.group(1)) <= 31
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"\b(1[5-9]\d\d|20\d\d)\s*[-–]\s*(1[5-9]\d\d|20\d\d|\d\d)\b(?![-/.]\d)",
            lambda m: (
                f"{m.group(1)} to {m.group(2)}"
                if len(m.group(2)) == 4 or int(m.group(2)) > int(m.group(1)[2:])
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"(?<![\d/])\b(\d{1,2})/((?:19|20)\d\d)\b",
            lambda m: (
                f"{MONTHS[int(m.group(1)) - 1]} {m.group(2)}"
                if 1 <= int(m.group(1)) <= 12
                else m.group(0)
            ),
            text,
        )
        set_phrases = {"24/7": "24 7", "50/50": "fifty fifty", "9/11": "nine eleven"}
        text = re.sub(
            r"(?<![\d/])\b(24/7|50/50|9/11)\b(?!/)",
            lambda m: set_phrases[m.group(1)],
            text,
        )
        text = re.sub(
            r"(?<![\d/.,])\b(\d+)\s+(\d{1,2})/(\d{1,2})\b(?!/\d)",
            lambda m: (
                f"{m.group(1)} and "
                f"{NumberVerbalizer.fraction_words(int(m.group(2)), int(m.group(3)), True)}"
                if 0 < int(m.group(2)) < int(m.group(3))
                and int(m.group(3)) in FRAC_DENOMINATORS
                else m.group(0)
            ),
            text,
        )

        def day_month_sub(match: re.Match[str]) -> str:
            first, second = int(match.group(1)), int(match.group(2))
            after = match.string[match.end() : match.end() + 12].lower()
            before = match.string[max(0, match.start() - 16) : match.start()]
            if re.match(r"\s*(stars?|points?|marks?|votes?)\b", after) or re.search(
                r"(rated|scored?|rating|got)\s*$", before.lower()
            ):
                return f"{first} out of {second}"
            if (
                0 < first < second
                and second in FRAC_DENOMINATORS
                and "0" not in (match.group(1)[0], match.group(2)[0])
                and (
                    not re.search(DATE_BEFORE_RE, before, re.I)
                    or re.match(FRAC_AFTER_RE, after)
                )
            ):
                return NumberVerbalizer.fraction_words(first, second)

            def ord_suffix(val: int) -> str:
                if 10 <= val % 100 <= 20:
                    return f"{val}th"
                return f"{val}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(val % 10, 'th') }"

            the = "" if re.search(r"\bthe\s+$", before, re.I) else "the "
            if 1 <= second <= 12 and 1 <= first <= 31:
                return f"{the}{ord_suffix(first)} of {MONTHS[second - 1]}"
            if 1 <= first <= 12 and 1 <= second <= 31:
                return f"{MONTHS[first - 1]} {ord_suffix(second)}"
            return match.group(0)

        text = re.sub(r"(?<![\d/])\b(\d{1,2})/(\d{1,2})\b(?!/\d)", day_month_sub, text)
        text = re.sub(
            r"\b(\d+(?:\.\d+){2,})(?:-(alpha|beta|rc\d*|preview))?\b",
            lambda m: (
                " point ".join(m.group(1).split("."))
                + (" " + m.group(2) if m.group(2) else "")
            ),
            text,
        )

        def plural_int(val_str: str, unit_word: str) -> str:
            val = int(val_str)
            return f"{val} {unit_word}{'' if val == 1 else 's'}"

        text = re.sub(
            r"\b(\d{1,2}):(\d{2}):(\d{2})\b(?!:)",
            lambda m: (
                f"{m.group(1)}:{m.group(2)} and {plural_int(m.group(3), 'second')}"
                if re.search(r"\b(at|by|until|from)\s+$", m.string[: m.start()])
                else ", ".join(
                    plural_int(x, w)
                    for x, w in zip(m.groups(), ("hour", "minute", "second"))
                    if int(x) or w == "second"
                )
            ),
            text,
        )
        text = re.sub(
            r"\b(\d{1,2}(?::\d{2})?\s?(?:[AaPp]\.?[Mm]\.?)?)\s?[-–]\s?"
            r"(\d{1,2}(?::\d{2})?\s?[AaPp]\.?[Mm]\b\.?)",
            r"\1 to \2",
            text,
        )
        text = re.sub(
            r"\b(\d{1,2}:\d{2})\s?[-–]\s?(\d{1,2}:\d{2})\b", r"\1 to \2", text
        )
        text = re.sub(r"\b(\d{1,2}):(\d)\b(?!\d)", r"\1 to \2", text)
        text = re.sub(
            r"(?<=[Rr]atio of |[Rr]atio is )(\d{1,2}):(\d{1,2})\b", r"\1 to \2", text
        )
        text = re.sub(
            r"\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6})*-\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b",
            lambda m: self._code_read(m.group(0)),
            text,
        )

        def phone_sub(match: re.Match[str]) -> str:
            raw = match.group(0)
            lead = "plus " if raw.strip().startswith("+") else ""
            groups = re.findall(r"\d+", raw)
            return (
                " " + lead + ", ".join(NumberVerbalizer.digits(g) for g in groups) + " "
            )

        text = re.sub(
            r"(?<![\w.,])\+?\(?\d{2,4}\)?(?:[\s.-]\(?\d{2,4}\)?){1,4}(?![\w.,]*\d)",
            lambda m: (
                phone_sub(m)
                if sum(ch.isdigit() for ch in m.group(0)) >= 7
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"\b((?:[Cc]all|[Dd]ial|[Tt]ext|[Rr]ing)\s+)(\d{3,6})\b",
            lambda m: m.group(1) + NumberVerbalizer.digits(m.group(2)),
            text,
        )

        text = re.sub(
            r"\b(\d+)\s?ft\.?\s?(\d{1,2})\s?in\b\.?",
            lambda m: (
                f"{m.group(1)} {'foot' if m.group(1) == '1' else 'feet'} "
                f"{m.group(2)} {'inch' if m.group(2) == '1' else 'inches'}"
            ),
            text,
        )
        text = re.sub(r"\b(\d)'\s?(\d{1,2})(?:\"|'')", r"\1 foot \2", text)
        unit_keys = "|".join(
            re.escape(k)
            for k in sorted(
                (k for k in UNITS if k not in ("in", "s")), key=len, reverse=True
            )
        )
        text = re.sub(
            UNIT_PREFIX_RE
            + r"\s?("
            + unit_keys
            + r")/("
            + unit_keys
            + r"|s|in)(?![\w/])",
            lambda m: (
                f"{m.group(1)} {self._format_unit(m.group(2), m.group(1))} "
                f"per {self._singularize_unit(UNITS.get(m.group(3), 'seconds'))}"
                if m.group(3) != "in"
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            UNIT_PREFIX_RE + r"\s?(" + unit_keys + r")(?![\w/'])",
            lambda m: f"{m.group(1)} {self._format_unit(m.group(2), m.group(1))}",
            text,
        )
        text = re.sub(
            r"(\b(?:squared|cubed|to the (?:minus )?(?:\w+ )?\w+(?:th|st|nd|rd))\s)"
            r"(" + unit_keys + r")(?:/(" + unit_keys + r"|s))?(?![\w/'])",
            lambda m: (
                m.group(1)
                + UNITS[m.group(2)]
                + (
                    " per " + self._singularize_unit(UNITS.get(m.group(3), "seconds"))
                    if m.group(3)
                    else ""
                )
            ),
            text,
        )

        text = re.sub(
            r"\b(" + ID_WORDS + r")(\s+(?:No\.?\s*|#\s*)?)([A-Z]{1,3}\s?)?(\d{2,4})"
            r"([A-Z]{0,2})\b(?![.,]\d|\s?[-–]\s?\d)",
            lambda m: (
                m.group(1)
                + m.group(2)
                + (NumberVerbalizer.spell(m.group(3)) + " " if m.group(3) else "")
                + NumberVerbalizer.number_read(m.group(4))
                + (" " + NumberVerbalizer.spell(m.group(5)) if m.group(5) else "")
            ),
            text,
        )
        text = re.sub(
            r"\b([A-Z]{2,3})\s(\d{3,4})\b(?![.,:]\d|\s\d)",
            lambda m: m.group(1) + " " + NumberVerbalizer.number_read(m.group(2)),
            text,
        )

        def allow_pair(match: re.Match[str]) -> bool:
            return not re.search(
                CODE_CONTEXT_RE,
                match.string[max(0, match.start() - 40) : match.start()],
            )

        text = re.sub(
            r"(?<![-/.\w])([A-Z]{1,3})(\d{1,4})([A-Z]{0,2})(?![-/\w])",
            lambda m: (
                " ".join(
                    part
                    for part in (
                        NumberVerbalizer.spell(m.group(1)),
                        NumberVerbalizer.number_read(m.group(2)),
                        NumberVerbalizer.spell(m.group(3)),
                    )
                    if part
                )
                if allow_pair(m)
                and (len(m.group(2)) <= 2 or (len(m.group(1)) == 2 and not m.group(3)))
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"(?<![-/.\w])(\d{1,4})([A-Z]{1,2})(?![-/\w])",
            lambda m: (
                NumberVerbalizer.number_read(m.group(1))
                + " "
                + NumberVerbalizer.spell(m.group(2))
                if allow_pair(m)
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"(?<![-/.\w])(?=[A-Za-z]*\d)(?:[A-Z][a-z]?\d{0,3}){2,}(?![-/\w])",
            lambda m: (
                " ".join(
                    NumberVerbalizer.spell(el)
                    + (" " + NumberVerbalizer.cardinal(int(cnt)) if cnt else "")
                    for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", m.group(0))
                )
                if all(el in ELEMENTS for el in re.findall(r"[A-Z][a-z]?", m.group(0)))
                and allow_pair(m)
                else m.group(0)
            ),
            text,
        )
        text = re.sub(
            r"#(\d+(?:-\d+)*)\b",
            lambda m: (
                "number "
                + ", ".join(
                    g if len(g) <= 3 else NumberVerbalizer.digits(g)
                    for g in m.group(1).split("-")
                )
            ),
            text,
        )
        text = re.sub(
            r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+"
            r"(?:-[A-Za-z0-9]+)+\b",
            lambda m: self._code_read(m.group(0)),
            text,
        )
        text = re.sub(
            r"\b(?=\w*\d)(?=\w*[A-Z])[A-Z0-9]{3,}\b",
            lambda m: NumberVerbalizer.spell(m.group(0)),
            text,
        )
        ordinal_or_unit = (
            r"\d+(?:st|nd|rd|th|am|pm|s|kg|km|mm|cm|ml|mg|gb|mb|tb|kb|ghz|mhz|kw|"
            r"kwh|mah|hrs?|mins?|h|m|k|x|g|l|w|v)"
        )
        text = re.sub(
            r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{4,}\b",
            lambda m: (
                m.group(0)
                if re.fullmatch(ordinal_or_unit, m.group(0), re.I)
                else NumberVerbalizer.spell(m.group(0))
            ),
            text,
        )
        text = re.sub(
            r"(?<![\d$€£.,])\b(\d{5,})\b(?![.,]\d)",
            lambda m: NumberVerbalizer.digits(m.group(1)),
            text,
        )
        text = re.sub(
            r"(?<![\d/:.-])\b(\d{1,4})\s*[-–]\s*(\d{1,4})\b(?![/:.-]\d)",
            r"\1 to \2",
            text,
        )
        text = re.sub(
            r"\b(\d+)\s*h\s*(\d{1,2})\b",
            lambda m: (
                f"{m.group(1)} hour{'' if m.group(1) == '1' else 's'} {m.group(2)}"
            ),
            text,
        )
        text = re.sub(
            r"\b(\d+)\s*hrs?\b",
            lambda m: f"{m.group(1)} hour{'' if m.group(1) == '1' else 's'}",
            text,
        )
        text = re.sub(
            r"\b(\d+)\s*mins?\b",
            lambda m: f"{m.group(1)} minute{'' if m.group(1) == '1' else 's'}",
            text,
        )
        text = re.sub(
            UNIT_PREFIX_RE + r"\s?(" + unit_keys + r")(?![\w/'])",
            lambda m: f"{m.group(1)} {self._format_unit(m.group(2), m.group(1))}",
            text,
        )
        text = re.sub(r"\s*×\s*", " by ", text)
        text = re.sub(r"(?<=\d)\s+x\s+(?=\d)", " by ", text)
        for pattern, replacement in SYMBOLS:
            text = re.sub(pattern, replacement, text)
        text = re.sub(
            r"£(\d[\d,]*)\.(\d{2})\b",
            lambda m: (
                f"{m.group(1)} pounds {int(m.group(2))}"
                if m.group(2) != "00"
                else f"{m.group(1)} pounds"
            ),
            text,
        )
        text = re.sub(r"\s+–\s+", " — ", text).replace("–", "-")
        return re.sub(r"\s{2,}", " ", text).strip()

    def __call__(self, text: str) -> str:
        """Normalizes ``text`` into spoken form."""
        out = self._normalize_pre(text)
        if re.search(r"[0-9#%$€£@&/_+=<>±≥≤×°]", out):
            if self._nemo is not None:
                try:
                    out = self._nemo.normalize(out, punct_post_process=True)
                except Exception:  # pylint: disable=broad-except
                    out = self._verbalizer.fallback(out)
            else:
                out = self._verbalizer.fallback(out)
        out = re.sub(r"\s+([.,!?;:])", r"\1", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        if self._respeller is not None:
            out = self._respeller(out)
        return out[:1].upper() + out[1:] if text[:1].isupper() else out
