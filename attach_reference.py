#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Tatiana Korol. Part of elan-asr-align.
# License: CC BY-NC 4.0 — non-commercial use only, with attribution (see LICENSE).
"""
attach_reference.py — put a linguist's transcription (with its glosses) onto the
real time of a recording.

Typical case: a field recording has been segmented and transcribed by an ASR
model (e.g. by asr_to_eaf.py: an .eaf with one tier of draft text), and there is the
linguist's own transcription of (part of) the same recording, made in ELAN,
Toolbox or FLEx, with words, glosses and notes but with dummy times
(0–4 s, 5–9 s …) or with no times at all. This script finds where each
reference sentence is spoken and writes an .eaf in which the reference tiers sit
on the real time, next to the ASR draft.

INPUT
  audio        the recording (wav / flac / mp3 / ogg; any sample rate, mono or stereo)
  reference    an .eaf whose top-level tier holds the sentences, with any
               dependent tiers (words, glosses, free translation, notes …),
               or a .txt file with one sentence per line
  --base EAF   optional: the .eaf to merge into (e.g. the ASR draft). Without it
               a new .eaf is written that links the audio and holds only the
               reference tiers.

HOW IT WORKS
  1. Emissions. The multilingual CTC alignment model MMS_FA (torchaudio) is run
     over the recording in 30-second chunks.
  2. Forced alignment. The reference text is romanised to the 26 Latin letters
     + apostrophe that MMS_FA knows (tones and diacritics dropped, IPA letters
     mapped: ŋ→ng, ɛ→e, ɓ→b …) and aligned with a "star" token at both ends.
     The star absorbs any audio that is not in the reference, so the reference
     may cover only part of the recording (the opening of a tale, one episode …).
     With --star-between the star is also allowed between sentences, for
     transcriptions that skip repetitions or asides.
  3. Boundaries. A sentence starts at its first word and ends at its last word.
     The boundary between two sentences is moved to the pause between them
     (energy-based speech/pause track from vad_core), or, if there is no pause,
     to the energy minimum. The outer edges are snapped to the speech onset/offset.
  4. Merge. All reference tiers are copied with their hierarchy; annotation and
     time-slot IDs are renumbered so nothing clashes with the base file. Words,
     glosses, notes in symbolic tiers follow their sentence automatically;
     time-aligned child tiers are stretched linearly from the old sentence
     interval to the new one. If a reference tier has the same name as a base
     tier, the BASE tier is renamed (default suffix "_ASR"): the linguist's tier
     names are kept.

OUTPUT
  the merged .eaf, a table of sentences with their times and an alignment score
  (mean CTC probability of the letters, 0–1; low = check by ear), and optionally
  a TSV of word times (--words-tsv).

EXAMPLE
  python attach_reference.py recording.mp3 manuscript.eaf \
      --base recording_asr.eaf -o recording_asr_ref.eaf

The MMS_FA weights (~1.2 GB, CC-BY-NC 4.0) are downloaded by torchaudio on
first use into $TORCH_HOME (default ~/.cache/torch).
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import vad_core
from common import MIME, SR, load_audio, new_eaf_root, set_media, write_eaf

HOP = 320                # wav2vec2 stride in samples → one CTC frame = 20 ms
FRAME_S = HOP / SR
# The first CTC letter of a word is placed on its acoustic peak, a frame or two
# after the real onset; a sentence start is never later than this before it.
WORD_MARGIN = 0.04
# A word whose letters got a mean CTC probability below this is listed in the
# report: usually the speaker said something else there, or the reference has a typo.
WEAK_WORD = 0.20

# Order of the top-level elements required by the EAF 3.0 schema.
EAF_ORDER = ["HEADER", "TIME_ORDER", "TIER", "LINGUISTIC_TYPE", "LOCALE",
             "LANGUAGE", "CONSTRAINT", "CONTROLLED_VOCABULARY", "LEXICON_REF",
             "EXTERNAL_REF"]


# ─────────────────────────── romanisation ────────────────────────────

_ROMAN = str.maketrans({
    # vowels
    "ɛ": "e", "ɔ": "o", "ə": "e", "ɨ": "i", "ʊ": "u", "ɪ": "i", "ɩ": "i",
    "ɐ": "a", "ʌ": "a", "æ": "ae", "ø": "o", "œ": "o", "ɵ": "o",
    "ɯ": "u", "ɤ": "o", "ɜ": "e", "ɑ": "a", "ɒ": "o",
    # consonants
    "ŋ": "ng", "ɲ": "ny", "č": "ch", "š": "sh", "ž": "zh",
    "ǯ": "j", "ǰ": "j", "ɣ": "g", "χ": "h", "ʁ": "r", "ħ": "h",
    "ʃ": "sh", "ʒ": "zh", "ʧ": "ch", "ʤ": "j", "θ": "t", "ð": "d",
    "β": "b", "ɸ": "f", "ɟ": "j", "ɡ": "g", "ʲ": "",
    "ɓ": "b", "ɗ": "d", "ƙ": "k", "ɠ": "g", "ʄ": "j", "ƴ": "y",
    "ɖ": "d", "ʈ": "t", "ɳ": "n", "ɽ": "r", "ʂ": "s", "ʐ": "z",
    "ɻ": "r", "ɹ": "r", "ɾ": "r", "ɺ": "l", "ɫ": "l", "ɬ": "l",
    "ɮ": "l", "ʎ": "y", "ʋ": "v", "ɦ": "h", "ç": "s", "ʝ": "j",
    "ɕ": "sh", "ʑ": "zh", "ɱ": "m", "ɴ": "n", "ʀ": "r", "ʙ": "b",
    "ʜ": "h", "ʟ": "l", "ɥ": "w", "ɰ": "w", "ʍ": "w", "ß": "ss",
    # length, stress, glottal stop: dropped
    "ː": "", "ˑ": "", "ʔ": "", "ʕ": "", "ˈ": "", "ˌ": "", "‿": "",
    # apostrophe-like letters → the apostrophe MMS_FA knows
    "ʼ": "'", "ʹ": "'", "’": "'", "‘": "'", "`": "'",
})


def romanize(word: str) -> str:
    """Word → letters a–z and ' only (the MMS_FA alphabet). '' for punctuation.

    The hyphen is removed on purpose: in the MMS_FA dictionary '-' is the CTC
    blank, and a blank inside the target sequence breaks forced alignment.
    """
    w = word.lower().translate(_ROMAN)
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = re.sub(r"[^a-z']", "", w).strip("'")
    return w


# ───────────────────────────── reference ─────────────────────────────

@dataclass
class Sentence:
    ann_id: str
    text: str
    old_t1: int | None
    old_t2: int | None
    words: list[str]              # original words that contain letters
    roman: list[str]              # their romanised forms (same length)
    label: str = ""
    start: float = 0.0            # aligned, seconds
    end: float = 0.0
    score: float = 0.0
    word_times: list[tuple[float, float, float]] = field(default_factory=list)


def eaf_from_lines(lines: list[str], tier_id: str = "Transcription") -> ET.Element:
    """A .txt reference (one sentence per line) → a one-tier EAF with dummy times."""
    root = new_eaf_root()
    to = root.find("TIME_ORDER")
    tier = ET.SubElement(root, "TIER", {"LINGUISTIC_TYPE_REF": "default-lt", "TIER_ID": tier_id})
    for i, line in enumerate(lines):
        a, b = f"ts{2 * i + 1}", f"ts{2 * i + 2}"
        ET.SubElement(to, "TIME_SLOT", {"TIME_SLOT_ID": a, "TIME_VALUE": str(i * 5000)})
        ET.SubElement(to, "TIME_SLOT", {"TIME_SLOT_ID": b, "TIME_VALUE": str(i * 5000 + 4000)})
        ann = ET.SubElement(ET.SubElement(tier, "ANNOTATION"), "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{i + 1}", "TIME_SLOT_REF1": a, "TIME_SLOT_REF2": b})
        ET.SubElement(ann, "ANNOTATION_VALUE").text = line
    ET.SubElement(root, "LINGUISTIC_TYPE", {"GRAPHIC_REFERENCES": "false",
                                            "LINGUISTIC_TYPE_ID": "default-lt",
                                            "TIME_ALIGNABLE": "true"})
    return root


def slot_values(root: ET.Element) -> dict[str, int | None]:
    out = {}
    for ts in root.iter("TIME_SLOT"):
        v = ts.get("TIME_VALUE")
        out[ts.get("TIME_SLOT_ID")] = int(v) if v not in (None, "") else None
    return out


def pick_sentence_tier(root: ET.Element, wanted: str | None) -> ET.Element:
    tiers = root.findall("TIER")
    if wanted:
        for t in tiers:
            if t.get("TIER_ID") == wanted:
                return t
        sys.exit(f"Tier {wanted!r} not found in the reference. Tiers: "
                 + ", ".join(t.get("TIER_ID") for t in tiers))
    for t in tiers:
        if t.get("PARENT_REF") is None and any(
                (a.findtext("ANNOTATION_VALUE") or "").strip()
                for a in t.iter("ALIGNABLE_ANNOTATION")):
            return t
    sys.exit("The reference has no top-level tier with time-aligned text; pass --ref-tier.")


def label_tier(root: ET.Element, sent_tier: ET.Element) -> dict[str, str]:
    """Short labels for the report (e.g. a 'Ref' tier with '1', '2a' …), if any."""
    sid = sent_tier.get("TIER_ID")
    for t in root.findall("TIER"):
        if t.get("PARENT_REF") != sid:
            continue
        vals = {a.get("ANNOTATION_REF"): (a.findtext("ANNOTATION_VALUE") or "").strip()
                for a in t.iter("REF_ANNOTATION")}
        if vals and all(len(v) <= 8 for v in vals.values()):
            return vals
    return {}


def read_sentences(root: ET.Element, tier: ET.Element) -> list[Sentence]:
    sv = slot_values(root)
    labels = label_tier(root, tier)
    out = []
    for a in tier.iter("ALIGNABLE_ANNOTATION"):
        text = (a.findtext("ANNOTATION_VALUE") or "").strip()
        words, roman = [], []
        for w in text.split():
            r = romanize(w)
            if r:
                words.append(w)
                roman.append(r)
        if not roman:
            continue
        out.append(Sentence(a.get("ANNOTATION_ID"), text,
                            sv.get(a.get("TIME_SLOT_REF1")), sv.get(a.get("TIME_SLOT_REF2")),
                            words, roman, labels.get(a.get("ANNOTATION_ID"), "")))
    # keep document order unless all old times are known
    if all(s.old_t1 is not None for s in out):
        out.sort(key=lambda s: s.old_t1)
    return out


# ───────────────────────────── alignment ─────────────────────────────

def load_model(device: str):
    import torch
    from torchaudio.pipelines import MMS_FA as bundle
    model = bundle.get_model(with_star=True).to(device)
    model.eval()
    return model, bundle.get_dict(star="*")


def compute_emissions(audio: np.ndarray, model, device: str,
                      chunk_s: float = 30.0, ctx_s: float = 2.0) -> np.ndarray:
    """CTC log-probabilities (T × V, last column = star) for the whole signal.

    Long recordings do not fit through wav2vec2 in one pass, so the signal is cut
    into chunks with `ctx_s` of context on each side; from every chunk only the
    frames of its central part are kept, placed by their absolute frame index.
    """
    import torch
    n = len(audio)
    total = n // HOP
    step = int(chunk_s * SR) // HOP * HOP
    ctx = int(ctx_s * SR) // HOP * HOP
    out = None
    filled = 0
    with torch.inference_mode():
        for s in range(0, n, step):
            a, b = max(0, s - ctx), min(n, s + step + ctx)
            if b - a < 800:
                break
            x = torch.from_numpy(audio[a:b])[None].to(device)
            em = model(x)[0][0].float().cpu().numpy()
            if out is None:
                out = np.empty((total, em.shape[1]), dtype=np.float32)
            g0, g1 = s // HOP, min(total, (s + step) // HOP)
            k0 = g0 - a // HOP
            part = em[k0:k0 + (g1 - g0)]
            out[g0:g0 + len(part)] = part
            filled = g0 + len(part)
            print(f"\r  emissions {min(b, n) / SR:7.1f} / {n / SR:.1f} s", end="", file=sys.stderr)
    print(file=sys.stderr)
    if out is None:
        sys.exit("The audio is too short.")
    if filled < total:                       # last few frames: repeat the last one
        out[filled:] = out[filled - 1]
    return out


def align(emission: np.ndarray, sents: list[Sentence], d: dict[str, int],
          offset_s: float, star_between: bool) -> None:
    """Forced alignment of all sentences at once; fills start/end/score/word_times."""
    import torch
    import torchaudio.functional as AF
    star = d["*"]
    targets: list[int] = []
    owner: list[tuple[int, int] | None] = []

    def put_star():
        targets.append(star)
        owner.append(None)

    put_star()
    for si, s in enumerate(sents):
        if star_between and si:
            put_star()
        for wi, w in enumerate(s.roman):
            for ch in w:
                targets.append(d[ch])
                owner.append((si, wi))
    put_star()

    if len(targets) * 2 > len(emission):
        sys.exit(f"The reference ({len(targets)} letters) is too long for "
                 f"{len(emission) * FRAME_S:.0f} s of audio.")

    ali, scores = AF.forced_align(torch.from_numpy(emission)[None],
                                  torch.tensor(targets, dtype=torch.int32)[None], blank=0)
    spans = AF.merge_tokens(ali[0], scores[0].exp(), blank=0)
    if len(spans) != len(targets):
        sys.exit("Internal error: token spans do not match the targets.")

    per_word: dict[tuple[int, int], list] = {}
    for sp, ow in zip(spans, owner):
        if ow is not None:
            per_word.setdefault(ow, []).append(sp)
    for si, s in enumerate(sents):
        s.word_times = []
        letters = []
        for wi in range(len(s.roman)):
            sp = per_word[(si, wi)]
            letters += [x.score for x in sp]
            s.word_times.append((offset_s + sp[0].start * FRAME_S,
                                 offset_s + sp[-1].end * FRAME_S,
                                 float(np.mean([x.score for x in sp]))))
        s.start, s.end = s.word_times[0][0], s.word_times[-1][1]
        s.score = float(np.mean(letters))


# ───────────────────────────── boundaries ────────────────────────────

def _pause_track(audio: np.ndarray, t0: float, t1: float, min_pause_ms: float):
    """Pauses [(start, end)] and an energy track in [t0, t1] (absolute seconds)."""
    a, b = max(0, int(t0 * SR)), min(len(audio), int(t1 * SR))
    info: dict = {}
    mask, hop, win = vad_core.speech_mask(audio[a:b], SR, close_ms=30.0,
                                          min_speech_ms=40.0, info=info)
    hop_s, half = hop / SR, win / SR / 2
    times = a / SR + half + np.arange(len(mask)) * hop_s
    pauses = []
    for r0, r1 in vad_core._runs(~mask):
        p0, p1 = times[r0], times[r1 - 1]
        if (p1 - p0) * 1000 >= min_pause_ms:
            pauses.append((float(p0), float(p1)))
    return pauses, times, mask, info["snr_track"]


def snap_boundaries(audio: np.ndarray, sents: list[Sentence], tol: float = 0.30,
                    min_pause_ms: float = 60.0, edge: float = 0.40) -> None:
    lo = max(0.0, sents[0].start - 3.0)
    hi = min(len(audio) / SR, sents[-1].end + 3.0)
    pauses, times, mask, snr = _pause_track(audio, lo, hi, min_pause_ms)

    def idx(t):
        return int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))

    # boundaries between neighbours
    for s, nxt in zip(sents, sents[1:]):
        e, b = s.end, nxt.start
        g0, g1 = min(e, b) - tol, max(e, b) + tol
        cand = [p for p in pauses if g0 <= (p[0] + p[1]) / 2 <= g1]
        if cand:
            p0, p1 = max(cand, key=lambda p: p[1] - p[0])
            # the energy track may call a weak onset (prevoicing of d, b …) or a
            # fading coda "pause"; never cut into a word the alignment has found
            s.end = max(p0, e)
            nxt.start = min(p1, b - WORD_MARGIN)
        else:                               # no pause: cut at the energy minimum
            i0, i1 = idx(min(e, b) - 0.10), idx(max(e, b) + 0.10)
            cut = float(times[i0 + int(np.argmin(snr[i0:i1 + 1]))])
            s.end = nxt.start = cut

    # outer edges: to the speech onset before the first word / offset after the last
    first_word, last_word = sents[0].start, sents[-1].end
    i = idx(sents[0].start)
    j = i
    if mask[i]:
        while j > 0 and mask[j - 1] and times[i] - times[j - 1] <= edge:
            j -= 1
    else:
        while j < len(mask) - 1 and not mask[j] and times[j] - times[i] <= edge:
            j += 1
        if not mask[j]:
            j = i
    sents[0].start = float(times[j])

    i = idx(sents[-1].end)
    j = i
    if mask[i]:
        while j < len(mask) - 1 and mask[j + 1] and times[j + 1] - times[i] <= edge:
            j += 1
    else:
        while j > 0 and not mask[j] and times[i] - times[j] <= edge:
            j -= 1
        if not mask[j]:
            j = i
    sents[-1].end = float(times[j])
    sents[0].start = min(sents[0].start, first_word - WORD_MARGIN)
    sents[-1].end = max(sents[-1].end, last_word)

    for s in sents:                         # 10 ms grid, never inverted
        s.start, s.end = round(s.start, 2), round(s.end, 2)
        if s.end <= s.start:
            s.end = s.start + 0.05


# ─────────────────────────────── merge ───────────────────────────────

def _num(x: str | None) -> int:
    m = re.search(r"(\d+)$", x or "")
    return int(m.group(1)) if m else 0


def _unique(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name}{k}" in taken:
        k += 1
    return f"{name}{k}"


def _reorder(root: ET.Element) -> None:
    kids = list(root)
    for k in kids:
        root.remove(k)
    rank = {t: i for i, t in enumerate(EAF_ORDER)}
    for k in sorted(kids, key=lambda e: rank.get(e.tag, len(rank))):
        root.append(k)


def merge(base: ET.Element, ref: ET.Element, sent_tier_id: str,
          sents: list[Sentence], base_suffix: str) -> dict[str, str]:
    """Copy all reference tiers into `base` at the new times. Returns {old→new tier id}
    for renamed BASE tiers (for the report)."""
    ref = copy.deepcopy(ref)
    sv = slot_values(ref)

    # 1. base tiers whose names clash with reference tiers get a suffix
    ref_ids = {t.get("TIER_ID") for t in ref.findall("TIER")}
    taken = ref_ids | {t.get("TIER_ID") for t in base.findall("TIER")}
    renamed = {}
    for t in base.findall("TIER"):
        tid = t.get("TIER_ID")
        if tid in ref_ids:
            new = _unique(tid + base_suffix, taken)
            taken.add(new)
            renamed[tid] = new
    for t in base.findall("TIER"):
        if t.get("TIER_ID") in renamed:
            t.set("TIER_ID", renamed[t.get("TIER_ID")])
        if t.get("PARENT_REF") in renamed:
            t.set("PARENT_REF", renamed[t.get("PARENT_REF")])

    # 2. linguistic types, constraints, languages, vocabularies …
    base_lts = {lt.get("LINGUISTIC_TYPE_ID"): lt for lt in base.findall("LINGUISTIC_TYPE")}
    lt_map = {}
    for lt in ref.findall("LINGUISTIC_TYPE"):
        lid = lt.get("LINGUISTIC_TYPE_ID")
        if lid in base_lts and dict(base_lts[lid].attrib) == dict(lt.attrib):
            continue
        if lid in base_lts:
            new = _unique(lid + "_ref", set(base_lts) | set(lt_map.values()))
            lt_map[lid] = new
            lt.set("LINGUISTIC_TYPE_ID", new)
        base.append(lt)
    for tag, key in (("CONSTRAINT", "STEREOTYPE"), ("LANGUAGE", "LANG_ID"),
                     ("CONTROLLED_VOCABULARY", "CV_ID"), ("LEXICON_REF", "LEX_REF_ID"),
                     ("EXTERNAL_REF", "EXT_REF_ID"), ("LOCALE", "LANGUAGE_CODE")):
        have = {e.get(key) for e in base.findall(tag)}
        for e in ref.findall(tag):
            if e.get(key) not in have:
                base.append(e)
                have.add(e.get(key))

    # 3. owner sentence of every alignable annotation, by its old midpoint
    sid_index = {s.ann_id: i for i, s in enumerate(sents)}
    old_iv = [(s.old_t1, s.old_t2) for s in sents]

    def owner_by_time(t1, t2):
        known = [t for t in (t1, t2) if t is not None]
        if not known:
            return None
        mid = sum(known) / len(known)
        for i, (a, b) in enumerate(old_iv):
            if a is not None and b is not None and a <= mid <= b:
                return i
        return min(range(len(sents)),
                   key=lambda i: min(abs(mid - (old_iv[i][0] or 0)), abs(mid - (old_iv[i][1] or 0))))

    def new_time(i: int, t: int | None) -> int | None:
        s = sents[i]
        if t is None:
            return None
        n1, n2 = int(round(s.start * 1000)), int(round(s.end * 1000))
        o1, o2 = s.old_t1, s.old_t2
        if o1 is None or o2 is None or o2 <= o1:
            return n1
        return int(round(n1 + (min(max(t, o1), o2) - o1) * (n2 - n1) / (o2 - o1)))

    # 4. new IDs
    hdr = base.find("HEADER")
    last_prop = next((p for p in hdr.findall("PROPERTY")
                      if p.get("NAME") == "lastUsedAnnotationId"), None)
    next_ann = max([_num(e.get("ANNOTATION_ID")) for e in base.iter() if e.get("ANNOTATION_ID")]
                   + [_num(last_prop.text) if last_prop is not None else 0, 0]) + 1
    next_ts = max([_num(t.get("TIME_SLOT_ID")) for t in base.iter("TIME_SLOT")] + [0]) + 1
    to = base.find("TIME_ORDER")
    if to is None:
        to = ET.Element("TIME_ORDER")
        base.insert(1, to)

    ann_map: dict[str, str] = {}
    for a in ref.iter():
        if a.tag in ("ALIGNABLE_ANNOTATION", "REF_ANNOTATION"):
            ann_map[a.get("ANNOTATION_ID")] = f"a{next_ann}"
            next_ann += 1

    slot_map: dict[tuple[int, str], str] = {}

    def slot(i: int, old_id: str) -> str:
        key = (i, old_id)
        if key not in slot_map:
            nonlocal next_ts
            sid = f"ts{next_ts}"
            next_ts += 1
            attrs = {"TIME_SLOT_ID": sid}
            nt = new_time(i, sv.get(old_id))
            if nt is not None:
                attrs["TIME_VALUE"] = str(nt)
            ET.SubElement(to, "TIME_SLOT", attrs)
            slot_map[key] = sid
        return slot_map[key]

    dropped = 0
    for t in ref.findall("TIER"):
        if t.get("LINGUISTIC_TYPE_REF") in lt_map:
            t.set("LINGUISTIC_TYPE_REF", lt_map[t.get("LINGUISTIC_TYPE_REF")])
        last_owner = 0
        for ann in list(t.findall("ANNOTATION")):
            a = ann[0]
            if a.tag == "ALIGNABLE_ANNOTATION":
                r1, r2 = a.get("TIME_SLOT_REF1"), a.get("TIME_SLOT_REF2")
                if t.get("TIER_ID") == sent_tier_id:
                    i = sid_index.get(a.get("ANNOTATION_ID"))
                    if i is None:                       # empty / punctuation-only sentence
                        t.remove(ann)
                        dropped += 1
                        continue
                    # new_time() maps the old edges exactly onto the aligned ones;
                    # slots are keyed per sentence, so a slot shared by two
                    # neighbouring sentences is split in two
                else:
                    i = owner_by_time(sv.get(r1), sv.get(r2))
                    i = last_owner if i is None else i
                last_owner = i
                a.set("TIME_SLOT_REF1", slot(i, r1))
                a.set("TIME_SLOT_REF2", slot(i, r2))
            for k in ("ANNOTATION_ID", "ANNOTATION_REF", "PREVIOUS_ANNOTATION"):
                if a.get(k) in ann_map:
                    a.set(k, ann_map[a.get(k)])
        base.append(t)

    # references to removed sentences would dangle: drop their dependants
    if dropped:
        alive = {a.get("ANNOTATION_ID") for a in base.iter() if a.get("ANNOTATION_ID")}
        changed = True
        while changed:
            changed = False
            for t in base.findall("TIER"):
                for ann in list(t.findall("ANNOTATION")):
                    ref_id = ann[0].get("ANNOTATION_REF")
                    if ref_id and ref_id not in alive:
                        alive.discard(ann[0].get("ANNOTATION_ID"))
                        t.remove(ann)
                        changed = True

    # 5. header bookkeeping, media types, element order
    if last_prop is None:
        last_prop = ET.SubElement(hdr, "PROPERTY", {"NAME": "lastUsedAnnotationId"})
    last_prop.text = str(next_ann - 1)
    for md in base.iter("MEDIA_DESCRIPTOR"):
        ext = Path(md.get("MEDIA_URL", "")).suffix.lower()
        if ext in MIME:
            md.set("MIME_TYPE", MIME[ext])
    _reorder(base)
    return renamed


def skeleton_from(ref: ET.Element) -> ET.Element:
    """An empty EAF that keeps the reference's header and types (no tiers, no times)."""
    base = copy.deepcopy(ref)
    for t in base.findall("TIER"):
        base.remove(t)
    to = base.find("TIME_ORDER")
    if to is not None:
        for ts in list(to):
            to.remove(ts)
    for p in base.find("HEADER").findall("PROPERTY"):
        if p.get("NAME") == "lastUsedAnnotationId":
            p.text = "0"
    return base


# ──────────────────────────────── main ───────────────────────────────

def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Put a linguist's transcription (EAF with glosses, or TXT) onto the real "
                    "time of a recording, optionally merging it into an ASR .eaf.")
    ap.add_argument("audio", type=Path)
    ap.add_argument("reference", type=Path, help=".eaf (top tier = sentences) or .txt (one per line)")
    ap.add_argument("-o", "--output", type=Path, help="output .eaf (default: <base or audio>_ref.eaf)")
    ap.add_argument("--base", type=Path, help=".eaf to merge into, e.g. the ASR draft")
    ap.add_argument("--ref-tier", help="reference tier with the sentences (default: first top-level tier)")
    ap.add_argument("--base-suffix", default="_ASR",
                    help="suffix for base tiers whose name clashes with a reference tier")
    ap.add_argument("--search", metavar="FROM:TO",
                    help="look for the reference only in this part of the audio, seconds (e.g. 0:120)")
    ap.add_argument("--star-between", action="store_true",
                    help="allow untranscribed audio between sentences too (skipped repetitions, asides)")
    ap.add_argument("--min-pause-ms", type=float, default=60.0)
    ap.add_argument("--warn-score", type=float, default=0.30,
                    help="sentences with a lower alignment score are marked 'check'")
    ap.add_argument("--words-tsv", type=Path, help="also write word times here")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # reference
    if args.reference.suffix.lower() == ".txt":
        lines = [x.strip() for x in args.reference.read_text(encoding="utf-8").splitlines() if x.strip()]
        ref = eaf_from_lines(lines, args.ref_tier or "Transcription")
    else:
        ref = ET.parse(args.reference).getroot()
    sent_tier = pick_sentence_tier(ref, args.ref_tier)
    sents = read_sentences(ref, sent_tier)
    if not sents:
        sys.exit("No text in the reference tier.")
    n_letters = sum(len(w) for s in sents for w in s.roman)
    print(f"Reference: tier {sent_tier.get('TIER_ID')!r}, {len(sents)} sentences, "
          f"{sum(len(s.roman) for s in sents)} words, {n_letters} letters", file=sys.stderr)

    # audio
    audio = load_audio(args.audio)
    t0, t1 = 0.0, len(audio) / SR
    if args.search:
        a, b = args.search.split(":")
        t0, t1 = float(a or 0), min(t1, float(b) if b else t1)
    print(f"Audio: {len(audio) / SR:.1f} s; searching {t0:.1f}–{t1:.1f} s on {device}", file=sys.stderr)

    model, d = load_model(device)
    seg = audio[int(t0 * SR):int(t1 * SR)]
    em = compute_emissions(seg, model, device)
    del model
    align(em, sents, d, t0, args.star_between)
    snap_boundaries(audio, sents, min_pause_ms=args.min_pause_ms)

    # report
    print(f"\n{'#':>3} {'ref':<5} {'start':>8} {'end':>8} {'score':>5}  text", file=sys.stderr)
    for i, s in enumerate(sents, 1):
        flag = "  ← check" if s.score < args.warn_score else ""
        txt = s.text if len(s.text) <= 60 else s.text[:57] + "…"
        print(f"{i:>3} {s.label:<5} {_fmt(s.start):>8} {_fmt(s.end):>8} {s.score:5.2f}  {txt}{flag}",
              file=sys.stderr)
        weak = [w for w, (_, _, sc) in zip(s.words, s.word_times) if sc < WEAK_WORD]
        if weak:
            print(f"{'':>34}weak words: {' '.join(weak)}", file=sys.stderr)

    # write
    if args.base:
        base = ET.parse(args.base).getroot()
        out = args.output or args.base.with_name(args.base.stem + "_ref.eaf")
    else:
        base = skeleton_from(ref)
        out = args.output or args.audio.with_name(args.audio.stem + "_ref.eaf")
        set_media(base, args.audio, out)
    renamed = merge(base, ref, sent_tier.get("TIER_ID"), sents, args.base_suffix)
    write_eaf(base, out)
    for old, new in renamed.items():
        print(f"Base tier {old!r} renamed to {new!r} (the reference uses that name).", file=sys.stderr)
    print(f"Written: {out}", file=sys.stderr)

    if args.words_tsv:
        with open(args.words_tsv, "w", encoding="utf-8") as f:
            f.write("sentence\tlabel\tword\tstart\tend\tscore\n")
            for i, s in enumerate(sents, 1):
                for w, (a, b, sc) in zip(s.words, s.word_times):
                    f.write(f"{i}\t{s.label}\t{w}\t{a:.2f}\t{b:.2f}\t{sc:.2f}\n")
        print(f"Word times: {args.words_tsv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
