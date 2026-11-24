#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr_to_eaf.py — cut a field recording into speech segments, transcribe every
segment with an Omnilingual ASR model and write an ELAN .eaf: one annotation
per segment, on one tier, linked to the recording.

STEPS
  1. Audio → 16 kHz mono (the loudest channel of a multichannel file).
  2. Speech/pause detection (vad_core.py: energy track, a noise floor that
     follows time, two thresholds relative to the recording's own dynamic
     range). Segments are split at pauses of at least --min-silence-ms, or at
     a pause length fitted to the recording (--auto-segment).
  3. Each segment → omniASR (--model, --lang), in batches. Segments longer than
     38 s are cut into pieces: the models accept at most 40 s per input.
  4. .eaf with one time-aligned tier (--tier, default "ASR").

The text is a DRAFT: unless the model was trained on your language, expect a
phonetic-looking approximation, not a transcription. The segmentation is
useful on its own (--keep-empty keeps segments the model returned nothing for).

To put an existing transcription onto the real time of the recording, use
attach_reference.py (it can merge into the .eaf written here).

EXAMPLE
  python asr_to_eaf.py recording.wav --lang xxx_Latn -o recording_asr.eaf

Requires the omnilingual-asr package: Linux (its fairseq2 backend also has
builds for macOS 14+ on Apple Silicon, but none for Windows; use WSL there).
Model weights are downloaded by fairseq2 on first use into $FAIRSEQ2_CACHE_DIR
(default ~/.cache/fairseq2/assets).
"""
from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import vad_core
from common import SR, load_audio, new_eaf_root, set_media, write_eaf

MODEL_MAX_S = 38.0      # the omniASR pipeline refuses inputs longer than 40 s
MODEL_MIN_S = 0.25      # the wav2vec2 feature extractor needs a minimum length


# ─────────────────────────────── EAF ─────────────────────────────────

def resolve_overlaps(anns: list[dict]) -> list[dict]:
    """Strictly non-overlapping, time-ordered annotations (one ELAN tier cannot
    hold overlaps). An overlap is split down the middle; a segment fully inside
    its predecessor is dropped."""
    out: list[dict] = []
    for ann in sorted((dict(a) for a in anns), key=lambda a: (a["start_ms"], a["end_ms"])):
        if ann["end_ms"] <= ann["start_ms"]:
            continue
        if out and ann["start_ms"] < out[-1]["end_ms"]:
            prev = out[-1]
            if ann["end_ms"] <= prev["end_ms"]:
                continue
            mid = (ann["start_ms"] + prev["end_ms"]) // 2
            if mid > prev["start_ms"]:
                prev["end_ms"] = mid
            ann["start_ms"] = max(mid, prev["end_ms"])
            if ann["end_ms"] <= ann["start_ms"]:
                continue
        out.append(ann)
    return out


def build_eaf(audio_path: Path, out: Path, anns: list[dict], tier_id: str) -> ET.Element:
    root = new_eaf_root()
    set_media(root, audio_path, out)
    anns = resolve_overlaps(anns)
    to = root.find("TIME_ORDER")
    values = sorted({ms for a in anns for ms in (a["start_ms"], a["end_ms"])})
    slot = {ms: f"ts{i + 1}" for i, ms in enumerate(values)}
    for ms in values:
        ET.SubElement(to, "TIME_SLOT", {"TIME_SLOT_ID": slot[ms], "TIME_VALUE": str(ms)})
    tier = ET.SubElement(root, "TIER", {"LINGUISTIC_TYPE_REF": "default-lt", "TIER_ID": tier_id})
    for i, a in enumerate(anns, 1):
        al = ET.SubElement(ET.SubElement(tier, "ANNOTATION"), "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{i}", "TIME_SLOT_REF1": slot[a["start_ms"]],
            "TIME_SLOT_REF2": slot[a["end_ms"]]})
        ET.SubElement(al, "ANNOTATION_VALUE").text = a["text"]
    ET.SubElement(root, "LINGUISTIC_TYPE", {"GRAPHIC_REFERENCES": "false",
                                            "LINGUISTIC_TYPE_ID": "default-lt",
                                            "TIME_ALIGNABLE": "true"})
    ET.SubElement(root.find("HEADER"), "PROPERTY",
                  {"NAME": "lastUsedAnnotationId"}).text = str(len(anns))
    return root


# ─────────────────────────────── ASR ─────────────────────────────────

def is_repetition_loop(text: str, min_repeats: int = 3) -> bool:
    """True if the text is one word n-gram repeated ≥ min_repeats times from the
    start ("la la la la …") — a typical hallucination of LLM-ASR on noise."""
    words = text.split()
    n = len(words)
    if n < min_repeats:
        return False
    for unit_len in range(1, n // min_repeats + 1):
        unit = words[:unit_len]
        repeats = 0
        for i in range(0, n, unit_len):
            if words[i:i + unit_len] == unit:
                repeats += 1
            else:
                break
        if repeats >= min_repeats:
            return True
    return False


def read_context(args) -> list[tuple[Path, str]]:
    """(audio, text) examples for the zero-shot model, from --context-tsv and
    --context-audio/--context-text pairs."""
    import soundfile as sf
    pairs: list[tuple[Path, str]] = []
    if args.context_tsv:
        base = args.context_tsv.resolve().parent
        for line in args.context_tsv.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.startswith("#") or "\t" not in line:
                continue
            p, text = line.split("\t", 1)
            p = Path(p.strip())
            pairs.append((p if p.is_absolute() else base / p, text.strip()))
    if len(args.context_audio) != len(args.context_text):
        sys.exit("--context-audio and --context-text must be given the same number of times.")
    pairs += list(zip(args.context_audio, args.context_text))

    out = []
    for p, text in pairs:
        if not text:
            continue
        if not p.exists():
            print(f"  context audio not found, skipped: {p}", file=sys.stderr)
            continue
        if sf.info(str(p)).duration > args.max_context_seconds:
            print(f"  context audio longer than {args.max_context_seconds:.0f} s, skipped: {p}",
                  file=sys.stderr)
            continue
        out.append((p, text))
        if len(out) >= args.max_context:
            break
    return out


def load_pipeline(args):
    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except ModuleNotFoundError:
        sys.exit("The omnilingual-asr package is not installed: pip install omnilingual-asr "
                 "(Linux; see README).")
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"Loading {args.model} on {device}…", file=sys.stderr)
    return ASRInferencePipeline(model_card=args.model, device=device, dtype=dtype)


def check_lang(args) -> None:
    """A wrong language code is a likely typo; stop early with suggestions."""
    is_llm = "_LLM_" in args.model
    if not is_llm:
        if args.lang:
            print(f"Note: {args.model} is a CTC model; --lang is ignored.", file=sys.stderr)
        return
    if not args.lang:
        print("Warning: no --lang given; LLM models transcribe worse without a language code.",
              file=sys.stderr)
        return
    try:
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
    except Exception:                                   # noqa: BLE001
        return
    if args.lang not in supported_langs:
        close = difflib.get_close_matches(args.lang, supported_langs, n=5, cutoff=0.6)
        sys.exit(f"--lang {args.lang!r} is not among the {len(supported_langs)} language codes "
                 f"the model knows (format: ISO 639-3 + script, e.g. eng_Latn)."
                 + (f" Close: {', '.join(close)}" if close else ""))


def transcribe(pipeline, paths: list[Path], args, context) -> list[str]:
    texts: list[str] = []
    bs = max(1, args.batch_size)
    for i in range(0, len(paths), bs):
        batch = [str(p) for p in paths[i:i + bs]]
        if context:
            from omnilingual_asr.models.inference.pipeline import ContextExample
            ctx = [ContextExample(audio=str(p), text=t) for p, t in context]
            out = pipeline.transcribe_with_context(batch, [ctx] * len(batch), batch_size=len(batch))
        else:
            lang = [args.lang] * len(batch) if args.lang else None
            out = pipeline.transcribe(batch, lang=lang, batch_size=len(batch))
        texts += [(t or "").strip() for t in out]
        print(f"\r  transcribed {len(texts)} / {len(paths)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return texts


# ──────────────────────────────── main ───────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cut a recording into speech segments, transcribe them with an "
                    "Omnilingual ASR model and write an ELAN .eaf.")
    ap.add_argument("audio", type=Path, help="wav / flac / mp3 / ogg (ffmpeg for others)")
    ap.add_argument("-o", "--output", type=Path, help="output .eaf (default: <audio>_asr.eaf)")

    g = ap.add_argument_group("model")
    g.add_argument("--model", default="omniASR_LLM_300M_v2",
                   help="omniASR model card, e.g. omniASR_CTC_1B_v2, omniASR_LLM_7B_v2, "
                        "omniASR_LLM_7B_ZS (default: %(default)s)")
    g.add_argument("--lang", help="language code for LLM models, ISO 639-3 + script: eng_Latn, fra_Latn, hau_Latn …")
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--device", help="cuda / cpu (default: cuda if available)")

    g = ap.add_argument_group("segmentation")
    g.add_argument("--min-silence-ms", type=float, default=500,
                   help="a pause at least this long ends a segment (default: %(default)s)")
    g.add_argument("--auto-segment", action="store_true",
                   help="fit the pause length to this recording instead of --min-silence-ms")
    g.add_argument("--min-speech-ms", type=float, default=200,
                   help="shorter sounds are ignored (default: %(default)s)")
    g.add_argument("--max-segment-s", type=float, default=15.0,
                   help="longer segments are split at their longest pause (default: %(default)s)")
    g.add_argument("--energy-threshold", type=float, default=0.0,
                   help="0 = adaptive (recommended); else an absolute RMS level, e.g. 0.05")
    g.add_argument("--pad-start-ms", type=float, default=vad_core.PAD_START_MS)
    g.add_argument("--pad-end-ms", type=float, default=vad_core.PAD_END_MS)

    g = ap.add_argument_group("output")
    g.add_argument("--tier", default="ASR", help="tier name (default: %(default)s)")
    g.add_argument("--keep-empty", action="store_true",
                   help="also write segments for which the model returned no text")

    g = ap.add_argument_group("context examples (only for omniASR_LLM_7B_ZS)")
    g.add_argument("--context-tsv", type=Path,
                   help="TSV without header: audio path <TAB> its transcription")
    g.add_argument("--context-audio", type=Path, action="append", default=[])
    g.add_argument("--context-text", action="append", default=[])
    g.add_argument("--max-context", type=int, default=10)
    g.add_argument("--max-context-seconds", type=float, default=30.0)
    args = ap.parse_args()

    t_start = time.time()
    out = args.output or args.audio.with_name(args.audio.stem + "_asr.eaf")

    zero_shot = args.model.split("@")[0].endswith("_ZS")
    context = read_context(args)
    if zero_shot and not context:
        sys.exit(f"{args.model} needs context examples: --context-tsv or --context-audio/--context-text.")
    if context and not zero_shot:
        sys.exit("Context examples work only with the zero-shot model omniASR_LLM_7B_ZS.")
    if not zero_shot:
        check_lang(args)

    # 1. audio
    audio = load_audio(args.audio)
    dur = len(audio) / SR
    print(f"Audio: {dur:.1f} s", file=sys.stderr)

    # 2. segments
    seg_kw = dict(min_speech_ms=args.min_speech_ms, max_seg_ms=args.max_segment_s * 1000,
                  pad_start_ms=args.pad_start_ms, pad_end_ms=args.pad_end_ms)
    if args.auto_segment:
        info: dict = {}
        segs = [(int(round(s)), int(round(e)))
                for s, e in vad_core.segment(audio, SR, profile="auto", info=info, **seg_kw)]
        print(f"Pause threshold fitted to the recording: {info.get('pause_threshold_ms', 0):.0f} ms",
              file=sys.stderr)
    else:
        segs = vad_core.detect_speech_segments(audio, SR, energy_threshold=args.energy_threshold,
                                               min_silence_ms=args.min_silence_ms, **seg_kw)
    if not segs:
        print("No speech detected; the whole recording is one segment.", file=sys.stderr)
        segs = [(0, int(dur * 1000))]
    speech = sum(e - s for s, e in segs) / 1000
    print(f"Segments: {len(segs)}, speech {speech / 60:.1f} min of {dur / 60:.1f} min", file=sys.stderr)

    # 3. segment files (long ones in ≤ 38 s pieces)
    import soundfile as sf
    tmp = Path(tempfile.mkdtemp(prefix="asr_to_eaf_"))
    try:
        pieces: list[tuple[int, int, Path | None]] = []
        for i, (s, e) in enumerate(segs):
            x = audio[int(s * SR / 1000):int(e * SR / 1000)]
            step = int(MODEL_MAX_S * SR)
            for k, off in enumerate(range(0, len(x), step)):
                part = x[off:off + step]
                p0 = s + int(off * 1000 / SR)
                p1 = s + int((off + len(part)) * 1000 / SR)
                if len(part) < MODEL_MIN_S * SR:
                    pieces.append((p0, p1, None))         # too short for the model
                    continue
                path = tmp / f"seg{i:05d}_{k}.wav"
                sf.write(str(path), part, SR)
                pieces.append((p0, p1, path))

        # 4. ASR
        pipeline = load_pipeline(args)
        todo = [p for _, _, p in pieces if p is not None]
        texts = dict(zip(todo, transcribe(pipeline, todo, args, context)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    anns, n_loops = [], 0
    for s, e, p in pieces:
        text = texts.get(p, "") if p is not None else ""
        if text and is_repetition_loop(text):
            n_loops += 1
            text = ""
        if text or args.keep_empty:
            anns.append({"start_ms": s, "end_ms": e, "text": text})
    if n_loops:
        print(f"Discarded {n_loops} repetition-loop outputs (\"la la la …\").", file=sys.stderr)

    # 5. EAF
    root = build_eaf(args.audio, out, anns, args.tier)
    write_eaf(root, out)
    n = len(root.find("TIER"))
    print(f"Written: {out} ({n} annotations, tier {args.tier!r}) in {time.time() - t_start:.0f} s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
