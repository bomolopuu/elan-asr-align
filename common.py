# -*- coding: utf-8 -*-
# Copyright (c) 2026 Tatiana Korol. Part of elan-asr-align.
# License: CC BY-NC 4.0 — non-commercial use only, with attribution (see LICENSE).
"""Shared helpers for asr_to_eaf.py and attach_reference.py: audio loading and
the bits of ELAN .eaf bookkeeping both scripts need."""
from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import vad_core

SR = 16000

XSI = "http://www.w3.org/2001/XMLSchema-instance"
ET.register_namespace("xsi", XSI)

MIME = {".wav": "audio/x-wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".mp4": "video/mp4",
        ".mpg": "video/mpeg", ".mov": "video/quicktime"}


def load_audio(path: Path) -> np.ndarray:
    """→ float32 mono 16 kHz. soundfile first, ffmpeg as a fallback (mp4, m4a …).

    From a multichannel file the loudest channel is taken, not the sum: in field
    recordings the second channel is often dead or noisy.
    """
    import soundfile as sf
    if not Path(path).is_file():
        sys.exit(f"Audio file not found: {path}")
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        if shutil.which("ffmpeg") is None:
            sys.exit(f"Cannot read {path} ({exc}); for this format install ffmpeg.")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                            "-ac", "1", "-ar", str(SR), str(tmp)], check=True)
            audio, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = vad_core.to_mono(audio)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if sr != SR:
        import torch
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), sr, SR).numpy()
    return audio


def new_eaf_root() -> ET.Element:
    """An empty EAF 3.0 document: ANNOTATION_DOCUMENT with HEADER and TIME_ORDER."""
    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "", "DATE": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "FORMAT": "3.0", "VERSION": "3.0",
        f"{{{XSI}}}noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan/EAFv3.0.xsd",
    })
    ET.SubElement(root, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})
    ET.SubElement(root, "TIME_ORDER")
    return root


def set_media(root: ET.Element, audio: Path, out: Path) -> None:
    """Link the recording: absolute URL plus a path relative to the output .eaf,
    so that ELAN finds the file when both are moved together."""
    hdr = root.find("HEADER")
    for md in hdr.findall("MEDIA_DESCRIPTOR"):
        hdr.remove(md)
    try:
        rel = os.path.relpath(audio.resolve(), out.resolve().parent).replace(os.sep, "/")
    except ValueError:                       # different drives on Windows
        rel = audio.name
    md = ET.Element("MEDIA_DESCRIPTOR", {
        "MEDIA_URL": audio.resolve().as_uri(),
        "MIME_TYPE": MIME.get(audio.suffix.lower(), "audio/x-wav"),
        "RELATIVE_MEDIA_URL": rel if rel.startswith(".") else "./" + rel,
    })
    hdr.insert(0, md)


def write_eaf(root: ET.Element, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, "    ")
    tree.write(out, encoding="UTF-8", xml_declaration=True)
