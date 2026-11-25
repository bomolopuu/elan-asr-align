# -*- coding: utf-8 -*-
# Copyright (c) 2026 Tatiana Korol. Part of elan-asr-align.
# License: CC BY-NC 4.0 — non-commercial use only, with attribution (see LICENSE).
"""
asr_formats.py — read ASR output with times from the usual formats into one
simple list of segments, and turn such a list into an ELAN tier.

Supported (the format is recognised by the content, not only by the extension):

  .eaf    ELAN: the first top-level tier with text (or the tier you name)
  .srt    SubRip:   1 / 00:00:11,580 --> 00:00:13,770 / text
  .vtt    WebVTT:   WEBVTT / 00:11.580 --> 00:13.770 / text
  .tsv    start <tab> end <tab> text   — seconds (11.58), or h:mm:ss(.ms);
  .txt    with or without a header row (start/begin, end/stop, text/transcription);
          a header like "start_ms" means milliseconds. Audacity label files
          (.txt, start <tab> end <tab> label) are read the same way.
  .json   Whisper / faster-whisper / WhisperX: {"segments": [{"start", "end",
          "text", "words": [{"word", "start", "end"}]}]};
          Hugging Face pipeline: {"chunks": [{"timestamp": [s, e], "text"}]};
          or a plain list of segment objects.

A plain text file without times is refused with an explanation: without times
the ASR cannot say where in the recording the text is.
"""
from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    start: float                     # seconds
    end: float
    text: str
    # word times, if the ASR gave them (Whisper JSON with word_timestamps)
    words: list[tuple[float, float, str]] = field(default_factory=list)


class AsrFormatError(SystemExit):
    pass


FORMATS_HELP = (
    "ASR must have times. Accepted: .eaf (ELAN), .srt, .vtt, a .tsv/.txt table "
    "'start<TAB>end<TAB>text' (seconds or h:mm:ss), or Whisper/faster-whisper/WhisperX/"
    "Hugging Face .json with segment times.")


# ───────────────────────────── times ─────────────────────────────────

_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:[.,]\d+)?)$")


def parse_time(s: str, ms: bool = False) -> float:
    """'11.58', '11,58', '00:11.580', '0:00:11,580', '1:02:03' → seconds."""
    s = s.strip()
    m = _TIME_RE.match(s)
    if not m:
        raise ValueError(f"not a time: {s!r}")
    a, b, c = m.groups()
    secs = float(c.replace(",", "."))
    if b is not None:                 # h:m:s
        secs += int(a) * 3600 + int(b) * 60
    elif a is not None:               # m:s
        secs += int(a) * 60
    elif ms:
        secs /= 1000
    return secs


# ──────────────────────────── readers ────────────────────────────────

def read_eaf(path: Path, tier_id: str | None = None) -> tuple[ET.Element, ET.Element, list[Segment]]:
    """→ (document, the ASR tier, its segments)."""
    root = ET.parse(path).getroot()
    tiers = root.findall("TIER")
    tier = None
    if tier_id:
        tier = next((t for t in tiers if t.get("TIER_ID") == tier_id), None)
        if tier is None:
            raise AsrFormatError(f"Tier {tier_id!r} not found in {path.name}. Tiers: "
                                 + ", ".join(t.get("TIER_ID") for t in tiers))
    else:
        tier = next((t for t in tiers if t.get("PARENT_REF") is None and any(
            (a.findtext("ANNOTATION_VALUE") or "").strip() for a in t.iter("ALIGNABLE_ANNOTATION"))), None)
        if tier is None:
            raise AsrFormatError(f"{path.name}: no top-level tier with time-aligned text.")
    sv = {}
    for ts in root.iter("TIME_SLOT"):
        v = ts.get("TIME_VALUE")
        sv[ts.get("TIME_SLOT_ID")] = int(v) / 1000 if v not in (None, "") else None
    segs = []
    for a in tier.iter("ALIGNABLE_ANNOTATION"):
        t1, t2 = sv.get(a.get("TIME_SLOT_REF1")), sv.get(a.get("TIME_SLOT_REF2"))
        text = (a.findtext("ANNOTATION_VALUE") or "").strip()
        if t1 is not None and t2 is not None and text:
            segs.append(Segment(t1, t2, text))
    return root, tier, segs


def read_subtitles(raw: str) -> list[Segment]:
    """SRT and WebVTT: a time line 'A --> B', then text lines up to a blank line."""
    segs: list[Segment] = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*(\S+)\s*-->\s*(\S+)", lines[i])
        if not m:
            i += 1
            continue
        t1, t2 = parse_time(m.group(1)), parse_time(m.group(2))
        i += 1
        text = []
        while i < len(lines) and lines[i].strip():
            text.append(lines[i].strip())
            i += 1
        txt = re.sub(r"<[^>]+>", "", " ".join(text))          # VTT/SRT styling tags
        txt = re.sub(r"\{[^}]*\}", "", txt).strip()             # SRT {\an8} codes
        if txt:
            segs.append(Segment(t1, t2, txt))
    return segs


def read_table(raw: str) -> list[Segment]:
    """start / end / text table, tab-separated (or ';' / ',' if there is no tab)."""
    lines = [l for l in raw.splitlines() if l.strip()]
    if not lines:
        return []
    delim = "\t" if "\t" in lines[0] else (";" if ";" in lines[0] else ",")
    rows = list(csv.reader(lines, delimiter=delim))
    head = [h.strip().lower() for h in rows[0]]
    ci_start, ci_end, ci_text, ms = 0, 1, 2, False
    try:
        parse_time(rows[0][0])
        has_header = False
    except (ValueError, IndexError):
        has_header = True

    def find(*names):
        for k, h in enumerate(head):
            if any(h == n or h.startswith(n) for n in names):
                return k
        return None

    if has_header:
        ci_start = find("start", "begin", "from", "tmin", "onset")
        ci_end = find("end", "stop", "to", "tmax", "offset")
        ci_text = find("text", "transcription", "transcript", "label", "value", "annotation", "sentence")
        if None in (ci_start, ci_end, ci_text):
            raise AsrFormatError(f"Table header {rows[0]} does not name start, end and text columns. "
                                 + FORMATS_HELP)
        ms = head[ci_start].endswith("ms") or "(ms)" in head[ci_start]
        rows = rows[1:]
    segs = []
    for r in rows:
        if len(r) <= max(ci_start, ci_end, ci_text):
            continue
        txt = r[ci_text].strip()
        if txt:
            segs.append(Segment(parse_time(r[ci_start], ms), parse_time(r[ci_end], ms), txt))
    return segs


def read_json(raw: str) -> list[Segment]:
    data = json.loads(raw)
    if isinstance(data, dict):
        if "segments" in data:
            items = data["segments"]
        elif "chunks" in data:                       # Hugging Face pipeline
            items = data["chunks"]
        else:
            raise AsrFormatError("JSON has neither 'segments' nor 'chunks'. " + FORMATS_HELP)
    elif isinstance(data, list):
        items = data
    else:
        raise AsrFormatError("Unexpected JSON. " + FORMATS_HELP)
    segs = []
    for it in items:
        if "timestamp" in it:                        # {"timestamp": [s, e], "text"}
            t1, t2 = it["timestamp"]
        else:
            t1, t2 = it.get("start"), it.get("end")
        txt = (it.get("text") or "").strip()
        if t1 is None or t2 is None or not txt:
            continue
        words = []
        for w in it.get("words") or []:              # word_timestamps / WhisperX
            if w.get("start") is not None and w.get("end") is not None:
                words.append((float(w["start"]), float(w["end"]), (w.get("word") or w.get("text") or "").strip()))
        segs.append(Segment(float(t1), float(t2), txt, words))
    return segs


def read_asr(path: Path, tier_id: str | None = None) -> tuple[ET.Element | None, list[Segment]]:
    """Any supported ASR file → (the .eaf document or None, segments in time order)."""
    if not path.exists():
        raise AsrFormatError(f"ASR file not found: {path}")
    if path.suffix.lower() == ".eaf":
        root, _, segs = read_eaf(path, tier_id)
        return root, segs
    raw = path.read_text(encoding="utf-8-sig")
    head = raw.lstrip()[:200]
    if head.startswith("<?xml") or head.startswith("<ANNOTATION_DOCUMENT"):
        root, _, segs = read_eaf(path, tier_id)
        return root, segs
    if head.startswith("{") or head.startswith("["):
        segs = read_json(raw)
    elif "-->" in raw:
        segs = read_subtitles(raw)
    else:
        first = [l for l in raw.splitlines() if l.strip()][:20]
        if not any(re.match(r"^\s*[\d:.,]+\s*[\t;,]", l) for l in first) and \
                not any(re.search(r"(?i)\b(start|begin)\b", l) for l in first[:1]):
            raise AsrFormatError(f"{path.name} looks like plain text without times. " + FORMATS_HELP)
        try:
            segs = read_table(raw)
        except ValueError:
            raise AsrFormatError(f"{path.name}: no times found. " + FORMATS_HELP)
    if not segs:
        raise AsrFormatError(f"{path.name}: no segments with times found. " + FORMATS_HELP)
    segs.sort(key=lambda s: s.start)
    return None, segs


# ──────────────────────────── segments → EAF ─────────────────────────

def eaf_from_segments(segs: list[Segment], tier_id: str = "ASR") -> ET.Element:
    """One time-aligned tier with the ASR segments. Overlapping segments (common
    in subtitles) are trimmed so that ELAN accepts them on one tier."""
    from common import new_eaf_root
    root = new_eaf_root()
    to = root.find("TIME_ORDER")
    tier = ET.SubElement(root, "TIER", {"LINGUISTIC_TYPE_REF": "default-lt", "TIER_ID": tier_id})
    last_end = 0
    n = 0
    for s in segs:
        t1 = max(int(round(s.start * 1000)), last_end)
        t2 = int(round(s.end * 1000))
        if t2 <= t1:
            continue
        n += 1
        ET.SubElement(to, "TIME_SLOT", {"TIME_SLOT_ID": f"ts{2 * n - 1}", "TIME_VALUE": str(t1)})
        ET.SubElement(to, "TIME_SLOT", {"TIME_SLOT_ID": f"ts{2 * n}", "TIME_VALUE": str(t2)})
        a = ET.SubElement(ET.SubElement(tier, "ANNOTATION"), "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{n}", "TIME_SLOT_REF1": f"ts{2 * n - 1}", "TIME_SLOT_REF2": f"ts{2 * n}"})
        ET.SubElement(a, "ANNOTATION_VALUE").text = s.text
        last_end = t2
    ET.SubElement(root, "LINGUISTIC_TYPE", {"GRAPHIC_REFERENCES": "false",
                                            "LINGUISTIC_TYPE_ID": "default-lt", "TIME_ALIGNABLE": "true"})
    ET.SubElement(root.find("HEADER"), "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(n)
    return root
