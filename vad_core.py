# -*- coding: utf-8 -*-
# Copyright (c) 2026 Tatiana Korol. Part of elan-asr-align.
# License: CC BY-NC 4.0 — non-commercial use only, with attribution (see LICENSE).
"""
vad_core.py — energy-based speech/pause detection for field recordings.

Used by asr_to_eaf.py (cutting a recording into segments) and by
attach_reference.py (moving sentence boundaries into pauses).

WHAT IT DOES (step by step)

  1. Mono. From a multichannel file the loudest channel is taken, not the sum:
     in field recordings the second channel is often dead or full of noise, and
     averaging halves the signal-to-noise ratio.

  2. Band-pass. The signal is limited to 80–5000 Hz by moving averages over a
     cumsum (O(n), no scipy), so that rumble, mains hum and table knocks do not
     lift the energy and "eat" quiet edges.

  3. Frame energy (25 ms window, 5 ms hop) in dB.

  4. A noise floor that FOLLOWS TIME: a percentile in a sliding ~3 s window
     instead of one global value. A fan, a change of distance to the microphone
     or moving around the room no longer break the threshold. (In the author's
     field recordings the noise floor differs by more than 50 dB between files.)

  5. Hysteresis — two thresholds. The core of a span is found with the high
     threshold, its edges are extended with the low one. Both are fractions of
     the recording's own dynamic range, not fixed numbers, so the same settings
     work on quiet and loud recordings. This keeps voiceless onsets and codas.

  6. Morphology: holes shorter than CLOSE_MS are filled, bursts shorter than
     MIN_SPEECH_MS are dropped.

  7. Phrase-pause threshold: either fixed (pause_ms) or fitted to the recording
     with Otsu's method over the LOGARITHM of the pause durations (pauses are
     roughly log-normal; on a linear scale a few long pauses pull the threshold
     up and glue phrases together).

  8. Short segments are merged into their neighbour; long ones are split at
     their longest internal pause.

  9. Padding from the edge of the voice (may be negative = narrowing).

The thresholds were tuned on hand-corrected segment boundaries of elicitation
and narrative recordings of West African languages; they have not been tuned
for other genres or recording set-ups.

Dependencies: numpy only.
"""

from __future__ import annotations

import math

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Defaults
# ─────────────────────────────────────────────────────────────────────────────

HOP_MS = 5.0        # frame hop
WIN_MS = 25.0       # frame length
BAND_LO = 80.0      # lower band edge, Hz
BAND_HI = 5000.0    # upper band edge, Hz

# Hysteresis thresholds as a FRACTION of the recording's dynamic range (from
# the noise floor to the speech level), with lower bounds in dB.
SNR_HI_FRAC = 0.65    # entering speech
SNR_LO_FRAC = 0.35    # extending the edges
SNR_HI_MIN_DB = 6.0
SNR_LO_MIN_DB = 3.0
RANGE_MIN_DB = 12.0   # bounds on the range estimate (degenerate recordings)
RANGE_MAX_DB = 45.0
SPEECH_PCT = 90       # percentile of frames taken as the speech level

FLOOR_WIN_S = 3.0     # window for tracking the noise floor
FLOOR_PCT = 10        # percentile within that window

MIN_SPEECH_MS = 150.0   # shorter than this is a click, not speech
CLOSE_MS = 150.0        # holes shorter than this are within a word; fill them
MIN_GAP_MS = 120.0      # shorter pauses are not candidates for phrase boundaries

MIN_SEG_MS = 250.0      # shorter segments are merged with the previous one
MAX_SEG_MS = 15000.0    # longer segments are split at their longest pause

# Bounds for the fitted (Otsu) pause threshold. A low ceiling suits elicitation,
# where phrases are separated by clear pauses; for long narratives use the
# 'coarse' profile.
PAUSE_MIN_MS = 200.0
PAUSE_MAX_MS = 400.0
PAUSE_FALLBACK_MS = 400.0

# Padding from the edge of the voice. These values were calibrated on a few
# hundred hand-corrected segment boundaries: after the hysteresis edges above,
# almost no padding is needed.
PAD_START_MS = 3.0
PAD_END_MS = -5.0

# Speech density above which a recording counts as "dense" (dialogue, narrative):
# there the fitted pause threshold must stay low, or turns get glued together.
DENSE_SPEECH = 0.60
DENSE_PAUSE_MAX_MS = 400.0

PROFILES = {
    # Phrases: the pause threshold is fitted to the recording.
    "auto": {},
    # Every acoustic block as its own span, no merging over pauses.
    "fine": {"pause_ms": 0.0, "min_seg_ms": 150.0},
    # Paragraphs/turns, for long narratives.
    "coarse": {"pause_ms": None, "pause_min_ms": 500.0, "pause_max_ms": 2500.0,
               "min_seg_ms": 800.0},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Signal helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_mono(audio, mode="loudest"):
    """Multichannel → one channel. By default the loudest channel is taken."""
    a = np.asarray(audio)
    if a.ndim == 1:
        return a.astype(np.float64, copy=False)
    if a.shape[1] == 1:
        return a[:, 0].astype(np.float64, copy=False)
    if mode == "mean":
        return a.mean(axis=1).astype(np.float64, copy=False)
    ch = int(np.argmax(a.std(axis=0)))
    return a[:, ch].astype(np.float64, copy=False)


def _movavg(x, n):
    """Moving average of length n via cumsum, zero-padded at the edges."""
    n = int(max(1, n))
    if n <= 1:
        return x
    half = n // 2
    xp = np.empty(len(x) + n, dtype=np.float64)
    xp[:half] = 0.0
    xp[half:half + len(x)] = x
    xp[half + len(x):] = 0.0
    c = np.cumsum(xp, out=xp)
    out = c[n - 1:n - 1 + len(x)].copy()
    out[1:] -= c[:len(x) - 1]
    out /= n
    return out


def bandpass(x, sr, lo_hz=BAND_LO, hi_hz=BAND_HI):
    """A coarse band-pass from moving averages (top: one smoothing; bottom:
    subtracting a double smoothing). Only meant to keep interference out of the
    frame energy, not to be an accurate filter."""
    y = np.asarray(x, dtype=np.float64)
    if hi_hz and hi_hz < sr / 2:
        n_hi = max(1, int(round(sr / (2.0 * hi_hz))))
        if n_hi > 1:
            y = _movavg(y, n_hi)
    if lo_hz and lo_hz > 0:
        n_lo = max(1, int(round(sr / lo_hz)))
        if n_lo > 1:
            y = y - _movavg(_movavg(y, n_lo), n_lo)
    return y


def n_samples(audio):
    """Length of the signal in samples."""
    return np.asarray(audio).shape[0]


def _channel(audio, mode="loudest"):
    """Index of the channel to use (None = average), from a thinned sample."""
    a = np.asarray(audio)
    if a.ndim == 1 or a.shape[1] == 1:
        return 0
    if mode == "mean":
        return None
    step = max(1, a.shape[0] // 200_000)
    return int(np.argmax(a[::step].astype(np.float64).std(axis=0)))


def _slice(audio, ch, i0, i1):
    a = np.asarray(audio)
    if a.ndim == 1:
        return a[i0:i1].astype(np.float64)
    if ch is None:
        return a[i0:i1].mean(axis=1).astype(np.float64)
    return a[i0:i1, ch].astype(np.float64)


def frame_db_track(audio, sr, hop_ms=HOP_MS, win_ms=WIN_MS,
                   band=(BAND_LO, BAND_HI), mono="loudest", chunk_s=120.0):
    """Frame energy in dB for the whole file, computed in chunks so that an
    hour-long recording does not need gigabytes of float64 temporaries.
    Returns (db, hop_samples, win_samples)."""
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    win = max(hop, int(round(sr * win_ms / 1000.0)))
    total = n_samples(audio)
    n = 1 + max(0, (total - win) // hop)
    if n < 2:
        return None, hop, win
    ch = _channel(audio, mono)
    margin = int(sr * 0.25) if band else 0
    per_chunk = max(1, int(chunk_s * 1000.0 / hop_ms))
    out = np.empty(n, dtype=np.float64)
    for k0 in range(0, n, per_chunk):
        k1 = min(n, k0 + per_chunk)
        s0, s1 = k0 * hop, (k1 - 1) * hop + win
        a0, a1 = max(0, s0 - margin), min(total, s1 + margin)
        x = _slice(audio, ch, a0, a1)
        if band:
            x = bandpass(x, sr, band[0], band[1])
        idx = np.arange(k0, k1) * hop - a0
        c = np.concatenate(([0.0], np.cumsum(x * x)))
        e = (c[idx + win] - c[idx]) / win
        out[k0:k1] = 10.0 * np.log10(np.maximum(e, 1e-20))
    return out, hop, win


def rolling_floor(db, hop_ms=HOP_MS, win_s=FLOOR_WIN_S, pct=FLOOR_PCT):
    """Noise floor that follows time: a percentile in a sliding window, computed
    at anchor points every half-window and interpolated linearly."""
    n = len(db)
    w = max(3, int(round(win_s * 1000.0 / hop_ms)))
    if n <= w:
        return np.full(n, float(np.percentile(db, pct)))
    step = max(1, w // 2)
    cs = list(range(0, n, step))
    if cs[-1] != n - 1:
        cs.append(n - 1)
    vals = [float(np.percentile(db[max(0, c - w // 2):min(n, c + w // 2)], pct))
            for c in cs]
    return np.interp(np.arange(n), cs, vals)


def _runs(mask):
    """Runs of True → [(start, end_exclusive)]."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def otsu(values):
    """Otsu threshold on a 1-D sample, O(n log n)."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n < 2:
        return float(v[0]) if n else 0.0
    cs = np.cumsum(v)
    k = np.arange(1, n)
    w0 = k / n
    w1 = 1.0 - w0
    m0 = cs[:-1] / k
    m1 = (cs[-1] - cs[:-1]) / (n - k)
    var = w0 * w1 * (m0 - m1) ** 2
    i = int(np.argmax(var))
    return float((v[i] + v[i + 1]) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
#  The speech / non-speech track
# ─────────────────────────────────────────────────────────────────────────────

def speech_mask(audio, sr, hop_ms=HOP_MS, win_ms=WIN_MS,
                band=(BAND_LO, BAND_HI), snr_hi_db=None, snr_lo_db=None,
                snr_hi_frac=SNR_HI_FRAC, snr_lo_frac=SNR_LO_FRAC,
                floor_win_s=FLOOR_WIN_S,
                min_speech_ms=MIN_SPEECH_MS, close_ms=CLOSE_MS,
                mono="loudest", info=None):
    """→ (mask, hop_samples, win_samples); mask[i] = frame i contains speech.

    Thresholds default to fractions of the recording's dynamic range; explicit
    snr_hi_db / snr_lo_db (dB above the noise floor) override them. If `info` is
    a dict, the SNR track and the measured levels are stored in it.
    """
    db, hop, win = frame_db_track(audio, sr, hop_ms, win_ms, band, mono)
    if db is None:
        return None, hop, win
    floor = rolling_floor(db, hop_ms, floor_win_s)
    snr = db - floor

    rng = float(np.percentile(snr, SPEECH_PCT))
    rng = min(max(rng, RANGE_MIN_DB), RANGE_MAX_DB)
    if snr_hi_db is None:
        snr_hi_db = max(SNR_HI_MIN_DB, snr_hi_frac * rng)
    if snr_lo_db is None:
        snr_lo_db = max(SNR_LO_MIN_DB, snr_lo_frac * rng)
    if snr_lo_db >= snr_hi_db:
        snr_lo_db = snr_hi_db * 0.6

    # hysteresis: runs above the low threshold that contain a frame above the high one
    core = snr > snr_hi_db
    ext = snr > snr_lo_db
    mask = np.zeros(len(db), dtype=bool)
    for a, b in _runs(ext):
        if core[a:b].any():
            mask[a:b] = True

    f_close = max(1, int(round(close_ms / hop_ms)))
    f_speech = max(1, int(round(min_speech_ms / hop_ms)))
    for a, b in _runs(~mask):
        if a > 0 and b < len(mask) and (b - a) < f_close:
            mask[a:b] = True
    for a, b in _runs(mask):
        if (b - a) < f_speech:
            mask[a:b] = False

    if info is not None:
        info["snr_track"] = snr
        info["floor_db"] = float(np.median(floor))
        info["range_db"] = rng
        info["snr_hi_db"] = float(snr_hi_db)
        info["snr_lo_db"] = float(snr_lo_db)
        info["speech_density"] = float(mask.mean())
    return mask, hop, win


# ─────────────────────────────────────────────────────────────────────────────
#  Phrase-pause threshold
# ─────────────────────────────────────────────────────────────────────────────

def pause_threshold(gaps_ms, speech_density=0.0,
                    lo=PAUSE_MIN_MS, hi=PAUSE_MAX_MS, info=None):
    """The pause length that separates phrases, fitted to the recording
    (Otsu over log durations, clipped to [lo, hi])."""
    g = [float(x) for x in gaps_ms if x >= MIN_GAP_MS]
    if speech_density >= DENSE_SPEECH:
        hi = min(hi, DENSE_PAUSE_MAX_MS)
    if len(g) < 3:
        thr = min(max(PAUSE_FALLBACK_MS, lo), hi)
    else:
        thr = math.exp(otsu(np.log(g)))
        thr = min(max(thr, lo), hi)
    if info is not None:
        info["pause_threshold_ms"] = float(thr)
        info["n_pauses"] = len(g)
    return float(thr)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def segment(audio, sr, profile="auto", pause_ms=None,
            min_seg_ms=MIN_SEG_MS, max_seg_ms=MAX_SEG_MS,
            pad_start_ms=PAD_START_MS, pad_end_ms=PAD_END_MS,
            pause_min_ms=PAUSE_MIN_MS, pause_max_ms=PAUSE_MAX_MS,
            info=None, **mask_kw):
    """Audio → [(start_ms, end_ms)] of speech segments.

    profile   'auto' (pause threshold fitted to the recording), 'fine' (no
              merging over pauses), 'coarse' (paragraphs). Explicit arguments
              override the profile.
    pause_ms  fixed pause length that separates segments; 0 = never merge;
              None = fit it to the recording.
    pad_*_ms  padding from the edge of the voice (negative = narrower).
    """
    prof = dict(PROFILES.get(profile, {}))
    if pause_ms is None and "pause_ms" in prof:
        pause_ms = prof["pause_ms"]
    min_seg_ms = prof.get("min_seg_ms", min_seg_ms)
    pause_min_ms = prof.get("pause_min_ms", pause_min_ms)
    pause_max_ms = prof.get("pause_max_ms", pause_max_ms)

    if info is None:
        info = {}
    hop_ms = mask_kw.get("hop_ms", HOP_MS)
    # a pause shorter than the hole-filling window must lower that window too,
    # or every pause_ms below CLOSE_MS would give the same result
    if pause_ms and 0 < pause_ms < mask_kw.get("close_ms", CLOSE_MS):
        mask_kw["close_ms"] = float(pause_ms)
    mask, hop, win = speech_mask(audio, sr, info=info, **mask_kw)
    if mask is None or not mask.any():
        return []
    info.pop("snr_track", None)

    rs = _runs(mask)
    gaps = [(rs[i + 1][0] - rs[i][1]) * hop_ms for i in range(len(rs) - 1)]
    if pause_ms is None:
        thr = pause_threshold(gaps, info.get("speech_density", 0.0),
                              pause_min_ms, pause_max_ms, info)
    else:
        thr = float(pause_ms)
        info["pause_threshold_ms"] = thr

    # merge neighbouring blocks across pauses shorter than the threshold
    merged = [list(rs[0])]
    for a, b in rs[1:]:
        if (a - merged[-1][1]) * hop_ms < thr:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    # merge too-short segments into the previous one
    out = []
    for a, b in merged:
        if out and (b - a) * hop_ms < min_seg_ms:
            out[-1][1] = b
        else:
            out.append([a, b])

    # split too-long segments at their longest internal pause
    res = []
    stack = list(reversed([tuple(o) for o in out]))
    while stack:
        a, b = stack.pop()
        if (b - a) * hop_ms <= max_seg_ms:
            res.append((a, b))
            continue
        inner = [(x, y) for (x, y) in _runs(~mask[a:b]) if x > 0 and y < b - a]
        if not inner:
            res.append((a, b))
            continue
        x, y = max(inner, key=lambda r: r[1] - r[0])
        mid = a + (x + y) // 2
        if mid <= a or mid >= b:
            res.append((a, b))
            continue
        stack.append((mid, b))
        stack.append((a, mid))
    res.sort()

    dur_ms = n_samples(audio) / sr * 1000.0
    segs = []
    for a, b in res:
        s = (a * hop + win / 2.0) / sr * 1000.0 + pad_start_ms
        e = ((b - 1) * hop + win / 2.0) / sr * 1000.0 + pad_end_ms
        s = max(0.0, s)
        e = min(dur_ms, e)
        if e - s >= 50.0:
            if segs and s < segs[-1][1]:          # never overlap
                s = segs[-1][1]
                if e - s < 50.0:
                    continue
            segs.append((round(s, 1), round(e, 1)))
    info["n_segments"] = len(segs)
    return segs


def detect_speech_segments(audio, sr, energy_threshold=0.0,
                           min_speech_ms=300, min_silence_ms=500, **kw):
    """[(start_ms, end_ms)] int, split at pauses of at least `min_silence_ms`.

    energy_threshold = 0 → adaptive threshold (recommended). A positive value
    is read as an absolute RMS level (0.005, 0.05 …) and converted into a
    threshold above this recording's measured noise floor.
    """
    snr_hi = kw.pop("snr_hi_db", None)
    if energy_threshold and energy_threshold > 0:
        # the noise floor for the conversion is measured on the UNFILTERED
        # signal, otherwise the numbers are not comparable
        db, _, _ = frame_db_track(audio, sr, band=None)
        if db is not None:
            floor = float(np.percentile(db, FLOOR_PCT))
            rng = float(np.percentile(db, SPEECH_PCT)) - floor
            rng = min(max(rng, RANGE_MIN_DB), RANGE_MAX_DB)
            snr_hi = 20.0 * math.log10(energy_threshold) - floor
            # stay below 80 % of the dynamic range, or a quiet phrase forms no
            # core at all and is lost entirely
            snr_hi = min(max(snr_hi, SNR_HI_MIN_DB), 0.8 * rng)
    return [(int(round(s)), int(round(e))) for s, e in
            segment(audio, sr, pause_ms=float(min_silence_ms),
                    min_speech_ms=float(min_speech_ms), snr_hi_db=snr_hi, **kw)]
