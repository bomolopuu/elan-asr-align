#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Tatiana Korol. Part of elan-asr-align.
# License: CC BY-NC 4.0 — non-commercial use only, with attribution (see LICENSE).
"""
ocr_asr_to_eaf.py — compare what OCR read from a handwritten page with what ASR
heard in the recording, and put both into one ELAN file on real time.

Typical case: a fieldworker wrote down a text by hand, with tone marks and an
interlinear gloss line under every line. The page has been read by OCR (a
handwriting model, a chatbot, or typed by hand), and the recording has been
transcribed by ASR (e.g. asr_to_eaf.py). OCR gives the right words and the
glosses but no times; ASR gives times but a rough text. This script brings them
together.

INPUT
  audio          the recording
  ocr            the OCR result, in one of these forms:
                   * chatbot-style text, one block per sentence:
                         1.
                         * Text: bə̀lò gó …          (also "Texte:", "Текст:")
                         * Gloss: | homme | avec | … (also "Gloses:", "Глоссы:")
                   * a .tsv with columns file, kind, text, where kind is
                     text / gloss / skip (as written by handwritten-interlinear-htr:
                     proof_page.py / recognize.py)
                   * a plain .txt, one sentence per line, no glosses
  --base EAF     the ASR .eaf (its first top-level tier is the ASR text)

HOW IT WORKS
  1. Compare. Every OCR sentence and the whole ASR text are romanised the same
     way (tones and diacritics dropped, ŋ→ng, ə→e …; spaces ignored, because
     ASR splits words differently). For each sentence the best-matching stretch
     of the ASR text is found (approximate substring search, edit distance).
     The report shows that stretch, its time and the character error rate (CER)
     between OCR and ASR. A high CER means: ASR misheard, OCR misread, or this
     sentence is not in the recording.
  2. Window. The matches tell where in the recording the page is spoken. Unless
     --search is given, only that part of the audio (plus --pad seconds) is
     used in the next step, which makes it much faster than searching the
     whole recording.
  3. Align. The OCR text is force-aligned to the audio with MMS_FA, exactly as
     in attach_reference.py (same romanisation, star token, pause snapping).
  4. Write. One .eaf: the ASR tier(s) of --base, plus the OCR tiers on real time:
         Transcription   OCR sentence                (time-aligned)
         Ref             sentence label: 1, 2, 2a …
         Words           one annotation per word     (symbolic subdivision)
         Gloss           one gloss per word          (when word and gloss counts agree)
         Gloss line      the whole gloss line        (when they do not agree)
         ASR match       "CER 23% · <ASR text of that stretch>"
     The ASR tier is renamed to "<name>_ASR" if it is also called
     "Transcription".

EXAMPLE
  python asr_to_eaf.py tale.wav --lang aal_Latn -o tale_asr.eaf
  python ocr_asr_to_eaf.py tale.wav page1_ocr.txt --base tale_asr.eaf -o tale_ocr_asr.eaf

The MMS_FA weights (~1.2 GB) are downloaded by torchaudio on first use into
$TORCH_HOME (default ~/.cache/torch). The comparison (step 1) needs no model:
with --compare-only the script stops after the report.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from attach_reference import (_fmt, align, compute_emissions, load_model, merge, pick_sentence_tier,
                              read_sentences, romanize, slot_values, snap_boundaries)
from common import SR, load_audio, new_eaf_root, write_eaf

PUNCT = set(".,;:!?«»\"'()[]…-–—/")


# ───────────────────────────── OCR input ─────────────────────────────

@dataclass
class OcrSentence:
    label: str
    text: str
    gloss: str | None = None
    # filled by compare()
    cer: float = 1.0
    t0: float = 0.0
    t1: float = 0.0
    asr_text: str = ""


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def read_ocr(path: Path) -> list[OcrSentence]:
    """Any supported OCR format → sentences with labels and (optional) gloss lines."""
    raw = path.read_text(encoding="utf-8-sig")
    out: list[OcrSentence] = []

    # a) table: file / kind / text  (handwritten-interlinear-htr)
    if path.suffix.lower() == ".tsv" or raw.startswith("file\tkind\ttext"):
        for r in csv.DictReader(raw.splitlines(), delimiter="\t"):
            kind = (r.get("kind") or "").strip()
            text = _clean(r.get("text") or "")
            if not text or kind == "skip":
                continue
            if kind == "gloss":
                if out and out[-1].gloss is None:
                    out[-1].gloss = text
            else:                                     # "text" (or the older name "ngen")
                out.append(OcrSentence(str(len(out) + 1), text))
        return out

    # b) chatbot style: "1." / "* Text: …" / "* Gloss: …"
    text_re = re.compile(r"^\s*[*\-•]?\s*(?:Text|Texte|Текст|Transcription)\s*:\s*(.*)$", re.I)
    gloss_re = re.compile(r"^\s*[*\-•]?\s*(?:Gloss(?:es)?|Gloses?|Глоссы)\s*:\s*(.*)$", re.I)
    num_re = re.compile(r"^\s*(\d+[a-z]?)\s*[.)]?\s*$")
    if any(text_re.match(l) for l in raw.splitlines()):
        label = None
        for line in raw.splitlines():
            if m := num_re.match(line):
                label = m.group(1)
            elif m := text_re.match(line):
                out.append(OcrSentence(label or str(len(out) + 1), _clean(m.group(1))))
                label = None
            elif (m := gloss_re.match(line)) and out and out[-1].gloss is None:
                out[-1].gloss = _clean(m.group(1))
        return out

    # c) plain text: one sentence per line
    return [OcrSentence(str(i + 1), _clean(l)) for i, l in enumerate(
        l for l in raw.splitlines() if l.strip())]


def words_of(text: str) -> list[str]:
    """Words that carry letters (punctuation tokens like ',' are left out)."""
    return [w for w in text.split() if romanize(w)]


def gloss_cells(gloss: str) -> list[str]:
    cells = [c.strip() for c in gloss.split("|")]
    return [c for c in cells if c and not all(ch in PUNCT for ch in c)]


# ─────────────────────────── ASR side ────────────────────────────────

@dataclass
class AsrStream:
    """All ASR text romanised into one string, with the source of every letter."""
    chars: str = ""
    times: list[float] = field(default_factory=list)      # estimated time of every letter
    word_ix: list[int] = field(default_factory=list)      # index into `words`
    words: list[str] = field(default_factory=list)


def asr_stream(base: ET.Element, tier_id: str | None) -> tuple[AsrStream, str]:
    tier = pick_sentence_tier(base, tier_id)
    sv = slot_values(base)
    st = AsrStream()
    chars: list[str] = []
    for a in tier.iter("ALIGNABLE_ANNOTATION"):
        t1, t2 = sv.get(a.get("TIME_SLOT_REF1")), sv.get(a.get("TIME_SLOT_REF2"))
        if t1 is None or t2 is None:
            continue
        romans = [(w, romanize(w)) for w in (a.findtext("ANNOTATION_VALUE") or "").split()]
        n = sum(len(r) for _, r in romans) or 1
        k = 0
        for w, r in romans:
            if not r:
                continue
            st.words.append(w)
            for ch in r:
                chars.append(ch)
                st.times.append((t1 + (t2 - t1) * (k + 0.5) / n) / 1000)   # spread letters evenly
                st.word_ix.append(len(st.words) - 1)
                k += 1
    st.chars = "".join(chars)
    return st, tier.get("TIER_ID")


def best_substring(pattern: str, text: str) -> tuple[int, int, int]:
    """Approximate substring search: the stretch text[start:end] with the smallest
    edit distance to `pattern`. Returns (distance, start, end). O(len(pattern)·len(text))
    with numpy rows."""
    m, n = len(pattern), len(text)
    t = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)

    def last_row(p: str, tt: np.ndarray) -> np.ndarray:
        idx = np.arange(len(tt) + 1)
        row = np.zeros(len(tt) + 1, dtype=np.int64)            # a match may start anywhere
        for i, ch in enumerate(p, 1):
            sub = row[:-1] + (tt != ord(ch))                    # match / substitute
            x = np.empty_like(row)
            x[0] = i
            x[1:] = np.minimum(row[1:] + 1, sub)                # or skip a pattern letter
            row = idx + np.minimum.accumulate(x - idx)          # or skip a text letter
        return row

    fwd = last_row(pattern, t)
    end = int(np.argmin(fwd))
    dist = int(fwd[end])
    # the start: same search with both strings reversed, ending at `end`
    rev = last_row(pattern[::-1], t[:end][::-1])
    start = end - int(np.argmin(rev))
    return dist, start, end


def compare(sents: list[OcrSentence], st: AsrStream) -> None:
    for s in sents:
        p = "".join(romanize(w) for w in s.text.split())
        if not p or not st.chars:
            continue
        dist, a, b = best_substring(p, st.chars)
        b = max(b, a + 1)
        s.cer = dist / len(p)
        s.t0, s.t1 = st.times[a], st.times[b - 1]
        w0, w1 = st.word_ix[a], st.word_ix[b - 1]
        s.asr_text = " ".join(st.words[w0:w1 + 1])


# ─────────────────────── reference EAF from OCR ───────────────────────

def ocr_eaf(sents: list[OcrSentence]) -> tuple[ET.Element, list[str]]:
    """OCR sentences → an EAF with dummy times (0–4 s, 5–9 s …) and the tiers
    Transcription / Ref / Words / Gloss (or Gloss line) / ASR match.
    attach_reference.merge() later moves everything onto the real times."""
    root = new_eaf_root()
    to = root.find("TIME_ORDER")
    tiers = {}

    def tier(tid, lt, parent=None):
        attrs = {"LINGUISTIC_TYPE_REF": lt, "TIER_ID": tid}
        if parent:
            attrs["PARENT_REF"] = parent
        tiers[tid] = ET.SubElement(root, "TIER", attrs)
        return tiers[tid]

    # order matters: attach_reference takes sentence labels from the first short child tier
    tier("Transcription", "utterance")
    tier("Ref", "ref", "Transcription")
    tier("Words", "words", "Transcription")
    tier("Gloss", "gloss", "Words")
    tier("Gloss line", "note", "Transcription")
    tier("ASR match", "note", "Transcription")

    n = 0
    warnings = []

    def new_id():
        nonlocal n
        n += 1
        return f"a{n}"

    def ref_ann(tid, parent_id, value, prev=None):
        aid = new_id()
        attrs = {"ANNOTATION_ID": aid, "ANNOTATION_REF": parent_id}
        if prev:
            attrs["PREVIOUS_ANNOTATION"] = prev
        a = ET.SubElement(ET.SubElement(tiers[tid], "ANNOTATION"), "REF_ANNOTATION", attrs)
        ET.SubElement(a, "ANNOTATION_VALUE").text = value
        return aid

    for i, s in enumerate(sents):
        ts1, ts2 = f"ts{2 * i + 1}", f"ts{2 * i + 2}"
        ET.SubElement(to, "TIME_SLOT", {"TIME_SLOT_ID": ts1, "TIME_VALUE": str(i * 5000)})
        ET.SubElement(to, "TIME_SLOT", {"TIME_SLOT_ID": ts2, "TIME_VALUE": str(i * 5000 + 4000)})
        sid = new_id()
        a = ET.SubElement(ET.SubElement(tiers["Transcription"], "ANNOTATION"), "ALIGNABLE_ANNOTATION",
                          {"ANNOTATION_ID": sid, "TIME_SLOT_REF1": ts1, "TIME_SLOT_REF2": ts2})
        ET.SubElement(a, "ANNOTATION_VALUE").text = s.text
        ref_ann("Ref", sid, s.label)

        words = words_of(s.text)
        cells = gloss_cells(s.gloss) if s.gloss else []
        prev = None
        word_ids = []
        for w in words:
            prev = ref_ann("Words", sid, w, prev)
            word_ids.append(prev)
        if s.gloss:
            if len(cells) == len(words):
                for wid, g in zip(word_ids, cells):
                    ref_ann("Gloss", wid, g)
            else:
                ref_ann("Gloss line", sid, s.gloss)
                warnings.append(f"sentence {s.label}: {len(words)} words but {len(cells)} gloss cells "
                                f"→ gloss kept as one line (tier 'Gloss line')")
        if s.asr_text:
            ref_ann("ASR match", sid, f"CER {s.cer:.0%} · {s.asr_text}")

    for lt_id, attrs in (
            ("utterance", {"TIME_ALIGNABLE": "true"}),
            ("ref", {"CONSTRAINTS": "Symbolic_Association", "TIME_ALIGNABLE": "false"}),
            ("words", {"CONSTRAINTS": "Symbolic_Subdivision", "TIME_ALIGNABLE": "false"}),
            ("gloss", {"CONSTRAINTS": "Symbolic_Association", "TIME_ALIGNABLE": "false"}),
            ("note", {"CONSTRAINTS": "Symbolic_Association", "TIME_ALIGNABLE": "false"})):
        ET.SubElement(root, "LINGUISTIC_TYPE", {"GRAPHIC_REFERENCES": "false",
                                                "LINGUISTIC_TYPE_ID": lt_id, **attrs})
    for stereo, desc in (("Symbolic_Subdivision", "Symbolic subdivision of a parent annotation."),
                         ("Symbolic_Association", "1-1 association with a parent annotation")):
        ET.SubElement(root, "CONSTRAINT", {"DESCRIPTION": desc, "STEREOTYPE": stereo})
    hdr = root.find("HEADER")
    ET.SubElement(hdr, "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(n)

    # drop tiers that stayed empty (e.g. no glosses at all)
    for tid, t in list(tiers.items()):
        if tid != "Transcription" and not len(t):
            root.remove(t)
    return root, warnings


# ──────────────────────────────── main ───────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare an OCR transcription of a handwritten page with the ASR of the "
                    "recording, align the OCR text to the audio and write one .eaf with both.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("audio", type=Path)
    ap.add_argument("ocr", type=Path, help="OCR result: chatbot-style .txt, .tsv (file/kind/text) or plain .txt")
    ap.add_argument("--base", type=Path, required=True, help="the ASR .eaf (e.g. from asr_to_eaf.py)")
    ap.add_argument("-o", "--output", type=Path, help="output .eaf (default: <base>_ocr.eaf)")
    ap.add_argument("--asr-tier", help="ASR tier in --base (default: first top-level tier with text)")
    ap.add_argument("--search", metavar="FROM:TO",
                    help="align only in this part of the audio, seconds (default: found from the ASR match)")
    ap.add_argument("--pad", type=float, default=10.0,
                    help="seconds added around the ASR match before aligning (default 10)")
    ap.add_argument("--max-cer", type=float, default=0.6,
                    help="if even the best OCR↔ASR matches are worse than this, search the whole audio")
    ap.add_argument("--compare-only", action="store_true", help="print the OCR↔ASR comparison and stop")
    ap.add_argument("--star-between", action="store_true",
                    help="allow audio between sentences that is not on the page (repetitions, asides)")
    ap.add_argument("--min-pause-ms", type=float, default=60.0)
    ap.add_argument("--warn-score", type=float, default=0.30,
                    help="sentences with a lower alignment score are marked 'check'")
    ap.add_argument("--words-tsv", type=Path, help="also write word times here")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 1. OCR
    sents = [s for s in read_ocr(args.ocr) if words_of(s.text)]
    if not sents:
        sys.exit(f"No sentences found in {args.ocr}.")
    print(f"OCR: {len(sents)} sentences, {sum(len(words_of(s.text)) for s in sents)} words, "
          f"{sum(1 for s in sents if s.gloss)} with glosses", file=sys.stderr)

    # 2. compare with ASR
    base = ET.parse(args.base).getroot()
    st, asr_tier = asr_stream(base, args.asr_tier)
    print(f"ASR: tier {asr_tier!r}, {len(st.words)} words", file=sys.stderr)
    compare(sents, st)
    print(f"\n{'ref':<5} {'ASR from':>8} {'to':>8} {'CER':>5}  OCR  /  ASR", file=sys.stderr)
    for s in sents:
        print(f"{s.label:<5} {_fmt(s.t0):>8} {_fmt(s.t1):>8} {s.cer:5.0%}  {s.text}", file=sys.stderr)
        print(f"{'':<29}{s.asr_text}", file=sys.stderr)
    good = [s for s in sents if s.cer <= args.max_cer]
    if args.compare_only:
        return 0

    # 3. which part of the audio to align in
    audio = load_audio(args.audio)
    dur = len(audio) / SR
    if args.search:
        a, b = args.search.split(":")
        t0, t1 = float(a or 0), min(dur, float(b) if b else dur)
    elif good:
        t0 = max(0.0, min(s.t0 for s in good) - args.pad)
        t1 = min(dur, max(s.t1 for s in good) + args.pad)
        print(f"\nThe page is spoken around {_fmt(t0 + args.pad)}–{_fmt(t1 - args.pad)} "
              f"(by the ASR match); aligning in {t0:.1f}–{t1:.1f} s.", file=sys.stderr)
    else:
        t0, t1 = 0.0, dur
        print(f"\nNo sentence matches the ASR better than CER {args.max_cer:.0%}: "
              f"searching the whole recording ({dur:.0f} s).", file=sys.stderr)

    # 4. OCR → reference EAF → forced alignment (attach_reference)
    ref, warnings = ocr_eaf(sents)
    for w in warnings:
        print(f"note: {w}", file=sys.stderr)
    sent_tier = pick_sentence_tier(ref, "Transcription")
    asents = read_sentences(ref, sent_tier)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, d = load_model(device)
    em = compute_emissions(audio[int(t0 * SR):int(t1 * SR)], model, device)
    del model
    align(em, asents, d, t0, args.star_between)
    snap_boundaries(audio, asents, min_pause_ms=args.min_pause_ms)

    by_label = {s.label: s for s in sents}
    print(f"\n{'ref':<5} {'start':>8} {'end':>8} {'align':>5} {'CER':>5}  text", file=sys.stderr)
    for s in asents:
        o = by_label.get(s.label)
        flag = "  ← check" if s.score < args.warn_score else ""
        txt = s.text if len(s.text) <= 55 else s.text[:52] + "…"
        print(f"{s.label:<5} {_fmt(s.start):>8} {_fmt(s.end):>8} {s.score:5.2f} "
              f"{(o.cer if o else 1):5.0%}  {txt}{flag}", file=sys.stderr)

    # 5. write
    out = args.output or args.base.with_name(args.base.stem + "_ocr.eaf")
    renamed = merge(base, ref, "Transcription", asents, "_ASR")
    write_eaf(base, out)
    for old, new in renamed.items():
        print(f"ASR tier {old!r} renamed to {new!r}.", file=sys.stderr)
    print(f"Written: {out}", file=sys.stderr)

    if args.words_tsv:
        with open(args.words_tsv, "w", encoding="utf-8") as f:
            f.write("sentence\tword\tstart\tend\tscore\n")
            for s in asents:
                for w, (a, b, sc) in zip(s.words, s.word_times):
                    f.write(f"{s.label}\t{w}\t{a:.2f}\t{b:.2f}\t{sc:.2f}\n")
        print(f"Word times: {args.words_tsv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
