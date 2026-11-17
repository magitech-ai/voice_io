"""Tests for text normalization, lexicon respelling, and local TTS runtime helpers."""

import builtins
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voice_io import TextToSpeech, to_pcm16
from voice_io.normalize import Normalizer, NumberVerbalizer
from voice_io.respell import Respeller
from voice_io.runtime import LocalInt8FlowBackend, write_wav
from voice_io.runtime import to_pcm16 as runtime_to_pcm16
from voice_io.tts import _resolve_weights_path


def test_number_verbalizer_forms() -> None:
    """Exercises digits, spell, cardinals, years, decimals, readouts, and fractions."""
    assert NumberVerbalizer.digits("1-2-09") == "one two zero nine"
    assert NumberVerbalizer.spell("A1!") == "A one"
    assert NumberVerbalizer.spell("ab-2") == "A B two"

    assert NumberVerbalizer.cardinal(0) == "zero"
    assert NumberVerbalizer.cardinal(1_002_005) == "one million two thousand and five"
    assert NumberVerbalizer.cardinal(1_000_000_000) == "one billion"

    assert NumberVerbalizer.year(2000) == "two thousand"
    assert NumberVerbalizer.year(2005) == "two thousand and five"
    assert NumberVerbalizer.year(1900) == "nineteen hundred"
    assert NumberVerbalizer.year(1905) == "nineteen oh five"
    assert NumberVerbalizer.year(1998) == "nineteen ninety eight"

    assert NumberVerbalizer.number("3.14") == "three point one four"
    assert NumberVerbalizer.number(".5") == "zero point five"
    assert NumberVerbalizer.number("42") == "forty two"

    assert NumberVerbalizer.number_read("007") == "zero zero seven"
    assert NumberVerbalizer.number_read("12345") == "one two three four five"
    assert NumberVerbalizer.number_read("400") == "four hundred"
    assert NumberVerbalizer.number_read("1200") == "twelve hundred"
    assert NumberVerbalizer.number_read("405") == "four oh five"
    assert NumberVerbalizer.number_read("1425") == "fourteen twenty five"
    assert NumberVerbalizer.number_read("19") == "nineteen"

    assert NumberVerbalizer.fraction_words(1, 2) == "one half"
    assert NumberVerbalizer.fraction_words(3, 2, lead=True) == "three halves"
    assert NumberVerbalizer.fraction_words(1, 4) == "one quarter"
    assert NumberVerbalizer.fraction_words(3, 5) == "three fifths"


def test_normalizer_money_time_dates_units_roman_and_codes() -> None:
    """Covers rule-based normalization paths across money, dates, units, and codes."""
    norm = Normalizer(use_nemo=False)

    assert "four point one two five dollars" in norm("$4.125")
    assert "one cent" in norm("$0.01").lower()
    assert "fifteen hundred yen" in norm("JPY 1500").lower()
    assert "ten to twenty dollars" in norm("$10-$20")

    assert "twelve o'clock" in norm("12:00")
    assert "fourteen hundred" in norm("14:00")
    assert "four o'clock" in norm("4:00")
    assert "four oh five" in norm("4:05")
    assert "five thirty pm" in norm("5:30 p.m.").lower()

    assert "nineteen nineties" in norm("1990s")
    assert norm("1995s") == "1995s"
    assert "two thousands" in norm("'00s")
    assert "nineties" in norm("90s")
    assert norm("'05s")

    assert "square meters" in norm("10 m^2")
    assert "three times faster" in norm("3x faster")
    assert norm("10x") == "10x"
    assert norm("10x20") == "one zero X two zero"
    assert "kilogram" in norm("1 kg")
    assert "five kilograms" in norm("5 kg")

    assert "four out of five stars" in norm("4/5 stars")
    assert "three quarters" in norm("3/4")
    assert "fifteenth of march" in norm("15/3").lower()
    assert "march fifteenth" in norm("3/15").lower()
    assert norm("99/99") == "ninety nine slash ninety nine"

    assert "chapter four" in norm("Chapter IV").lower()
    assert "parts two through three" in norm("Parts II through III").lower()
    assert "henry the eighth" in norm("Henry VIII").lower()
    assert "pope benedict the sixteenth" in norm("Pope Benedict XVI").lower()
    assert "king charles the third" in norm("King Charles III").lower()

    out = norm("Ping @alice_hq, @open_ai, @devOpsTeam, foo_bar_baz, HTTP2, xkcd, bcdf")
    assert "alice" in out.lower()
    assert "foo underscore bar underscore baz" in out.lower()
    assert "two" in out.lower()
    assert "fifty pounds" in norm("£50")
    assert "one euro and twenty cents" in norm("€1.20")
    assert "fifty percent" in norm("50%")
    assert "march fifteenth" in norm("2024-03-15").lower()
    assert "first second third" in norm("1st 2nd 3rd 4th 21st")

    for extra_in in [
        "(415) 555-0199",
        "+1-800-555-0199",
        "192.168.1.1",
        "user@example.com",
        "https://example.com/a/b",
        "24/7",
        "9/11",
        "50/50",
        "2^4",
        "10 m²",
        "1.5x faster",
        "speed 2x",
        "-12°C",
        "72°F",
        "10-20%",
        "5-10",
        "2020-2024",
        "1,000,000",
        "2 + 3 = 5",
        "16:9",
        "NASA",
        "SQL",
        "BCDF",
        "ABC-123",
        "UTF-8",
        "25/12/2024",
        "12/25/2024",
        "99/99/2024",
        "31/4",
        "0/5",
        "World War II",
        "Chapters I-V",
        "Pope John Paul II",
        "Title I",
        "Vol. II",
        "Part IV",
        "Act V",
        "Louis XIV",
        "1 ms",
        "1 ft",
        "USD 1",
        "$1.00",
        "$1.01",
        "€1",
        "£1",
        "$10–$20",
        "1900's",
        "'80s",
        "10s",
        "12:00 noon",
        "00:00",
        "0:05",
        "@bot123",
        "@a_dev",
        "gpt4o",
        "x86_64",
        "C++",
        "C#",
        "v1.2.3",
        "e.g. test, i.e. fine, etc.",
        "Mr. Kim met Dr. Jones.",
    ]:
        assert norm(extra_in)


def test_normalizer_nemo_integration_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies optional NeMo backend loading, execution, and error fallback."""
    orig_import = builtins.__import__

    def _missing_nemo(
        name: str,
        globals_: Any = None,
        locals_: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("nemo_text_processing"):
            raise ImportError("no nemo")
        return orig_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_nemo)
    norm_missing = Normalizer(use_nemo=True)
    assert "forty two" in norm_missing("42")

    calls: list[str] = []

    class _FakeNemoNormalizer:
        def __init__(self, input_case: str, lang: str) -> None:
            del input_case, lang

        def normalize(self, text: str, verbose: bool = False) -> str:
            del verbose
            calls.append(text)
            if "BOOM" in text:
                raise RuntimeError("nemo failed")
            return "nemo normalized output"

    fake_pkg = types.ModuleType("nemo_text_processing")
    fake_tn = types.ModuleType("nemo_text_processing.text_normalization")
    fake_norm = types.ModuleType("nemo_text_processing.text_normalization.normalize")
    setattr(fake_pkg, "__path__", [])
    setattr(fake_pkg, "text_normalization", fake_tn)
    setattr(fake_tn, "__path__", [])
    setattr(fake_tn, "normalize", fake_norm)
    setattr(fake_norm, "Normalizer", _FakeNemoNormalizer)

    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "nemo_text_processing", fake_pkg)
    monkeypatch.setitem(sys.modules, "nemo_text_processing.text_normalization", fake_tn)
    monkeypatch.setitem(
        sys.modules,
        "nemo_text_processing.text_normalization.normalize",
        fake_norm,
    )

    norm_with_nemo = Normalizer(use_nemo=True)
    assert norm_with_nemo("hello 42")
    fallback_out = norm_with_nemo("BOOM 42")
    assert "forty two" in fallback_out


def test_respeller_lexicon_edge_cases(tmp_path: Path) -> None:
    """Verifies missing/empty lexicons, requires_caps, and hyphen rejection."""
    missing = Respeller(str(tmp_path / "missing.tsv"))
    assert missing("Siobhan") == "Siobhan"

    empty_path = tmp_path / "empty.tsv"
    empty_path.write_text("# comment\n\n", encoding="utf-8")
    empty = Respeller(str(empty_path))
    assert empty("Siobhan") == "Siobhan"

    lex_path = tmp_path / "lex.tsv"
    lex_path.write_text(
        "api\tay pee eye\nNICE\tnyss\tc\ncache\tkash\n",
        encoding="utf-8",
    )
    resp = Respeller(str(lex_path))
    assert resp("NICE day vs nice day") == "Nyss day vs nice day"
    assert resp("Cache hit") == "Kash hit"
    assert resp("cache hit") == "kash hit"
    assert resp("API-first") == "Ay pee eye-first"
    assert resp("pre-cache") == "pre-cache"


def test_runtime_local_int8_backend_helpers_and_wav_roundtrip(
    tmp_path: Path,
) -> None:
    """Verifies runtime voice errors, voices property, call API, int8 mode, and WAV."""
    weights_path = _resolve_weights_path()
    with pytest.raises(ValueError):
        LocalInt8FlowBackend(weights_path, voice="nonexistent")

    backend = LocalInt8FlowBackend(weights_path, dec_mode="int8")
    assert "juno" in backend.voices
    assert "mira" in backend.voices

    tts = TextToSpeech(backend=backend, retries=0)
    tts_wav = tts("Hi.", seed=42)
    assert len(tts_wav) > 0

    backend.seed(42)
    wav_arr = backend("Hi.")
    assert isinstance(wav_arr, np.ndarray)
    assert wav_arr.dtype == np.float32
    assert len(wav_arr) > 0

    lat = backend.latents("Hi.")
    forced_latent = backend.speech.generate(
        backend.voice,
        backend.speech.encode_text("Hi."),
        max_frames=2,
        force=lat[:2],
    )
    assert forced_latent.shape[0] >= 1

    pcm_cls = LocalInt8FlowBackend.to_pcm16(
        np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    )
    pcm = to_pcm16(np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32))
    assert len(pcm) == 10 and pcm_cls == pcm
    assert to_pcm16(np.zeros(0, dtype=np.float32)) == b""
    assert len(to_pcm16(np.array([10, -10], dtype=np.int16))) == 4
    assert len(to_pcm16(np.array([0.1, -0.1], dtype=np.float64))) == 4
    assert len(to_pcm16(np.array([[0.1, 0.2]], dtype=np.float32))) == 4

    out_wav = tmp_path / "helper.wav"
    LocalInt8FlowBackend.write_wav(str(out_wav), wav_arr)
    assert out_wav.exists()

    out_wav_2 = tmp_path / "module_helper.wav"
    write_wav(str(out_wav_2), wav_arr)
    assert out_wav_2.exists()

    rt_pcm = runtime_to_pcm16(np.array([-0.5, 0.5], dtype=np.float32))
    assert len(rt_pcm) == 4


def test_normalizer_uncovered_branches() -> None:
    """Covers pronounceability, handles, Roman numerals, units, schemes, and streets."""
    norm = Normalizer()

    assert norm._is_pronounceable("abcdn") is False
    assert norm._is_pronounceable("bright") is True
    assert norm._code_read("AB--12") == "A B 12"

    samples = [
        "Build bright-12 and abcdn-12 and main-4-beta and AB-12-NASA.",
        "Follow @openai and @devxhq on stage.",
        "Test ID, walk to IV, Zone DX, Toy Story IV, Batman IV, Pope Francis II, word II.",
        "Dose is 1 mmHg for pressure.",
        "CAD 12.50 and AUD 5.00 and $12.50-$15.75 today.",
        "Duration 1h 05m or 2m 01s and wait 1m in the 00s.",
        "Math 2^n plus x^2 plus y^3 equals zero.",
        "Links git:// and s3://bucket.local and my-host://test.org here.",
        "Edit src/my-app_v1.conf and foo/bar.yaml soon.",
        "123A North Main St. Next block is 45 Oak Ave.",
        "Arrive at 10:30:01 and depart from 09:05:45.",
        "Gate B12 and code A12 are ready.",
    ]
    outputs = [norm(text) for text in samples]
    assert all(isinstance(out, str) and len(out) > 0 for out in outputs)
    assert "at open" in outputs[1]
    assert "millimeters of mercury" in outputs[3]
    assert "squared" in outputs[6] and "cubed" in outputs[6]
    assert "Street" in outputs[9] and "Avenue" in outputs[9]
