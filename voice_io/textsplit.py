"""Utterance chunking, Markdown sanitization, and LLM token stream buffering."""

import os
import re

DEFAULT_CLAUSE_MAX = 120
DEFAULT_FIRST_MAX = int(os.environ.get("VOICE_IO_FIRST_MAX", "0"))
FIRST_MIN = 20
MIN_PIECE = 30
_CONJ_PATTERN = (
    r"\b(?:and|but|or|nor|that|which|who|whom|whose|when|where|while|"
    r"because|although|though|so|if|as|than|until)\b"
)

_FENCED_CODE_RE = re.compile(r"```[^\n]*\n?[\s\S]*?```", re.MULTILINE)
_IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$")
_BLOCKQUOTE_RE = re.compile(r"(?m)^\s{0,3}>\s?")
_BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+")
_BOLD_ITALIC_STAR_RE = re.compile(r"\*{1,3}([^*\n]+?)\*{1,3}")
_BOLD_ITALIC_UNDER_RE = re.compile(r"(?<!\w)_{1,3}([^_\n]+?)_{1,3}(?!\w)")
_STRIKE_RE = re.compile(r"~~([^~\n]+?)~~")
_HRULE_RE = re.compile(r"(?m)^\s*(?:[-*_]){3,}\s*$")

_ABBREV_END_RE = re.compile(
    r"(?:\b(?:e\.g|i\.e|a\.m|p\.m|vs|etc|Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.|" r"\.\d+)$",
    re.IGNORECASE,
)


class SpeechTextSanitizer:
    """Converts LLM-formatted Markdown text into clean, speakable prose."""

    def __init__(
        self,
        *,
        code_block_cue: str = "",
    ) -> None:
        """Initializes the sanitizer.

        Args:
          code_block_cue: Optional spoken phrase substituted for fenced code blocks
            (for example, ``"Code snippet omitted."``). Empty omits blocks quietly.
        """
        self._code_block_cue = code_block_cue.strip()

    def clean(self, text: str) -> str:
        """Strips Markdown markup while preserving prose, numbers, and bare URLs.

        Args:
          text: Raw text or LLM Markdown response.

        Returns:
          Speakable plain text ready for spoken-form normalization and TTS.
        """
        if not text or not text.strip():
            return ""

        cue = f" {self._code_block_cue} " if self._code_block_cue else " "
        out = _FENCED_CODE_RE.sub(cue, text)
        out = _IMAGE_LINK_RE.sub(lambda m: m.group(1), out)
        out = _MD_LINK_RE.sub(lambda m: m.group(1), out)
        out = _INLINE_CODE_RE.sub(lambda m: m.group(1), out)
        out = _HEADING_RE.sub(
            lambda m: (
                m.group(1)
                if m.group(1).endswith((".", "!", "?", ":", "…"))
                else f"{m.group(1)}."
            ),
            out,
        )
        out = _BLOCKQUOTE_RE.sub("", out)
        out = _BULLET_RE.sub("", out)
        out = _HRULE_RE.sub(" ", out)
        out = _BOLD_ITALIC_STAR_RE.sub(lambda m: m.group(1), out)
        out = _BOLD_ITALIC_UNDER_RE.sub(lambda m: m.group(1), out)
        out = _STRIKE_RE.sub(lambda m: m.group(1), out)
        out = re.sub(r"\n+", " ", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out


class TextSplitter:
    """Splits input text into synthesis-friendly sentences and clauses."""

    def __init__(
        self,
        clause_max: int = DEFAULT_CLAUSE_MAX,
        first_max: int = DEFAULT_FIRST_MAX,
    ) -> None:
        """Initializes the text splitter.

        Args:
          clause_max: Maximum character length before splitting at clause boundaries.
          first_max: Optional tighter limit for the first utterance (0 disables).
        """
        self._clause_max = clause_max
        self._first_max = first_max

    @staticmethod
    def _split_conjunctions(text: str, clause_max: int) -> list[str]:
        """Splits an overlong clause without punctuation near a middle conjunction."""
        if len(text) <= clause_max:
            return [text]
        candidates = [
            match.start() for match in re.finditer(r"\s(?=" + _CONJ_PATTERN + ")", text)
        ]
        candidates = [
            idx
            for idx in candidates
            if idx >= MIN_PIECE and len(text) - idx >= MIN_PIECE
        ]
        if not candidates:
            return [text]
        cut = min(candidates, key=lambda idx: abs(idx - len(text) / 2))
        return TextSplitter._split_conjunctions(
            text[:cut].strip(), clause_max
        ) + TextSplitter._split_conjunctions(text[cut:].strip(), clause_max)

    @staticmethod
    def _split_first(text: str, first_max: int) -> list[str]:
        """Splits the first utterance at a clause boundary within ``first_max``."""
        if len(text) <= first_max:
            return [text]
        cuts = [match.end() for match in re.finditer(r"[,;:](?=\s)|—|--", text)]
        quotes = [idx for idx, ch in enumerate(text) if ch in '"“”']
        cuts = [
            cut
            for cut in cuts
            if cut >= FIRST_MIN
            and len(text) - cut >= FIRST_MIN
            and not any(abs(q_idx - cut) < 12 for q_idx in quotes)
        ]
        if not cuts:
            return [text]
        inside = [cut for cut in cuts if cut <= first_max]
        cut = max(inside) if inside else min(cuts)
        return [text[:cut].strip(), text[cut:].strip()]

    def split(self, text: str) -> list[str]:
        """Splits ``text`` into sentences and bounded clauses.

        Args:
          text: Input normalized text.

        Returns:
          List of non-empty utterance pieces.
        """
        out: list[str] = []
        for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
            if not sentence:
                continue
            if len(sentence) <= self._clause_max:
                out.append(sentence)
                continue
            clauses = [
                clause
                for clause in re.split(r"(?<=[,;:])\s+|(?<=—)\s*|(?<=--)\s*", sentence)
                if clause
            ]
            merged: list[str] = []
            current = ""
            for clause in clauses:
                if current and len(current) + 1 + len(clause) > self._clause_max:
                    merged.append(current)
                    current = clause
                elif not current:
                    current = clause
                elif current.endswith(("—", "--")):
                    current = current + clause
                else:
                    current = current + " " + clause
            if current:
                merged.append(current)
            for item in merged:
                out.extend(self._split_conjunctions(item, self._clause_max))
        if out and self._first_max:
            out = self._split_first(out[0], self._first_max) + out[1:]
        return out


class StreamingSentenceBuffer:
    """Accumulates streamed LLM tokens and emits complete speakable sentences."""

    def __init__(
        self,
        *,
        sanitizer: SpeechTextSanitizer | None = None,
        min_chars: int = 12,
    ) -> None:
        """Initializes the streaming sentence buffer.

        Args:
          sanitizer: Optional Markdown-to-speech sanitizer applied to emitted units.
          min_chars: Minimum characters before flushing on punctuation boundaries.
        """
        self._sanitizer = sanitizer
        self._min_chars = min_chars
        self._buf = ""

    @staticmethod
    def _has_open_fence(text: str) -> bool:
        """Returns True while an odd number of triple-backtick fences are open."""
        return text.count("```") % 2 == 1

    @staticmethod
    def _find_split_index(text: str, min_chars: int) -> int | None:
        """Returns the end index of a safe sentence or paragraph boundary."""
        para = text.find("\n\n")
        if para >= min_chars:
            return para + 2

        for match in re.finditer(r"([.!?…])(?=\s|\n|$)", text):
            end = match.end()
            if end < min_chars:
                continue
            # Wait for trailing context at end of buffer so decimals/extensions finish.
            if end == len(text):
                continue
            prefix = text[:end]
            if match.group(1) == "." and _ABBREV_END_RE.search(prefix):
                continue
            return end
        return None

    def _format_piece(self, piece: str) -> str | None:
        """Sanitizes an emitted chunk and drops empty results."""
        cleaned = self._sanitizer.clean(piece) if self._sanitizer else piece.strip()
        return cleaned if cleaned else None

    def push(self, delta: str) -> list[str]:
        """Appends a streamed text delta and returns any complete sentences."""
        if not delta:
            return []
        self._buf += delta
        if self._has_open_fence(self._buf):
            return []

        emitted: list[str] = []
        while True:
            split_at = self._find_split_index(self._buf, self._min_chars)
            if split_at is None:
                break
            raw_piece = self._buf[:split_at]
            self._buf = self._buf[split_at:].lstrip()
            formatted = self._format_piece(raw_piece)
            if formatted:
                emitted.append(formatted)
        return emitted

    def flush(self) -> list[str]:
        """Flushes any remaining buffered text at end-of-stream."""
        if not self._buf:
            return []
        remainder = self._buf
        self._buf = ""
        if self._has_open_fence(remainder):
            remainder = remainder.split("```", 1)[0]
        formatted = self._format_piece(remainder)
        return [formatted] if formatted else []


def split_text(
    text: str,
    clause_max: int = DEFAULT_CLAUSE_MAX,
    first_max: int = DEFAULT_FIRST_MAX,
) -> list[str]:
    """Convenience function wrapping :class:`TextSplitter`."""
    return TextSplitter(clause_max=clause_max, first_max=first_max).split(text)
