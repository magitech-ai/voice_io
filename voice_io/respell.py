"""Respellings for hard names and foreign words.

Matches entries from ``respell.tsv`` case-insensitively and accent-insensitively
on whole words so an English-letter TTS voice pronounces them naturally (e.g.,
``Siobhan -> Shivawn``, ``gnocchi -> nyocky``).
"""

import os
import re
import unicodedata
from collections.abc import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEXICON_PATH = os.path.join(_HERE, "respell.tsv")
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


class Respeller:
    """Dictionary-based respeller backed by a tab-separated lexicon."""

    def __init__(self, path: str = DEFAULT_LEXICON_PATH) -> None:
        """Initializes the respeller from a TSV lexicon file.

        Args:
          path: Path to the respelling TSV file (``written<TAB>respelling<TAB>flags``).
        """
        self._lexicon: dict[str, tuple[str, bool]] = {}
        self._max_words = 1
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                key = self.fold(parts[0])
                requires_caps = "c" in (parts[2] if len(parts) > 2 else "")
                self._lexicon[key] = (parts[1], requires_caps)
                self._max_words = max(self._max_words, key.count(" ") + 1)

    @staticmethod
    def fold(text: str) -> str:
        """Normalizes a lookup key by stripping accents and folding hyphens.

        Args:
          text: Input word or phrase.

        Returns:
          Lowercase ASCII-folded lookup key.
        """
        normalized = unicodedata.normalize(
            "NFKD", text.replace("’", "'").replace("‘", "'")
        )
        stripped = "".join(
            ch for ch in normalized if not unicodedata.combining(ch)
        ).lower()
        for src, dst in (
            ("ł", "l"),
            ("ø", "o"),
            ("æ", "ae"),
            ("œ", "oe"),
            ("ß", "ss"),
            ("đ", "d"),
        ):
            stripped = stripped.replace(src, dst)
        return " ".join(re.sub(r"[-‐–]", " ", stripped).split())

    def lookup(self, source: str) -> tuple[tuple[str, bool] | None, str]:
        """Looks up a word or phrase, preserving a trailing possessive ``'s``.

        Args:
          source: Original surface text span.

        Returns:
          A tuple ``((respelling, requires_caps), suffix)`` or ``(None, "")``.
        """
        key = self.fold(source)
        hit = self._lexicon.get(key)
        if hit:
            return hit, ""
        if key.endswith("'s") and key[:-2] in self._lexicon:
            return self._lexicon[key[:-2]], source[-2:]
        return None, ""

    @staticmethod
    def hyphen_ok(text: str, start: int, end: int, source: str) -> bool:
        """Checks whether a hyphen-adjacent match should be respelled alone.

        A hyphen-joined word matches alone only when it is a capitalized name and
        the adjacent part is lowercase (e.g., ``Worcestershire-glazed``).

        Args:
          text: Full text string.
          start: Start index of the candidate span.
          end: End index of the candidate span.
          source: Substring ``text[start:end]``.

        Returns:
          True if the hyphen context allows respelling.
        """
        left_hyphen = (
            text[start - 1 : start] == "-" and text[start - 2 : start - 1].isalpha()
        )
        right_hyphen = text[end : end + 1] == "-" and text[end + 1 : end + 2].isalpha()
        checks: tuple[tuple[bool, Callable[[], list[str]]], ...] = (
            (left_hyphen, lambda: text[: start - 1].split()[-1:]),
            (right_hyphen, lambda: text[end + 1 :].split()[:1]),
        )
        for has_hyphen, neighbor_fn in checks:
            if has_hyphen:
                other = (neighbor_fn() or [""])[0]
                if not (source[:1].isupper() and other[:1].islower()):
                    return False
        return True

    def __call__(self, text: str) -> str:
        """Respells matching names and foreign words in ``text``.

        Args:
          text: Input text.

        Returns:
          Text with matched spans replaced by their phonetic respellings.
        """
        if not self._lexicon:
            return text
        text = text.replace("’", "'")
        words = list(_WORD_RE.finditer(text))
        out: list[str] = []
        pos = 0
        idx = 0
        while idx < len(words):
            for count in range(min(self._max_words, len(words) - idx), 0, -1):
                span = words[idx : idx + count]
                if any(
                    text[left.end() : right.start()] not in (" ", "-")
                    for left, right in zip(span, span[1:])
                ):
                    continue
                start, end = span[0].start(), span[-1].end()
                source = text[start:end]
                if not self.hyphen_ok(text, start, end, source):
                    continue
                hit, tail = self.lookup(source)
                if not hit:
                    continue
                replacement, requires_caps = hit
                if requires_caps and not all(
                    match.group(0)[:1].isupper() for match in span
                ):
                    continue
                if source[:1].isupper() and replacement[:1].islower():
                    replacement = replacement[:1].upper() + replacement[1:]
                out.append(text[pos:start] + replacement + tail)
                pos = end
                idx += count
                break
            else:
                idx += 1
        return "".join(out) + text[pos:]
