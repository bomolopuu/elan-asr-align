# elan-asr-align

**Author:** Tatiana Korol ([ORCID 0009-0005-6590-2624](https://orcid.org/0009-0005-6590-2624))
· **License:** [CC BY-NC 4.0](LICENSE) — free for non-commercial use, with
attribution ([how to credit](#licenses)).

Two command-line tools for field linguists who work in [ELAN](https://archive.mpi.nl/tla/elan):

1. **`asr_to_eaf.py`** — cut a recording into speech segments and transcribe
   every segment with Meta's multilingual
   [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
   (omniASR, 1600+ languages). Output: an `.eaf` with one draft-text tier.
2. **`attach_reference.py`** — take a transcription you already have (an `.eaf`
   with words, glosses and translations on dummy times, or a plain `.txt`), find
   where each sentence is spoken, and write an `.eaf` in which all its tiers sit
   on the real time of the recording — optionally merged into the ASR `.eaf`
   from step 1.

They are meant for recordings of languages that have little or no ASR of their
own, where a rough segmentation and a draft are a start, and where a checked
transcription often exists on paper, in Toolbox/FLEx or in an `.eaf` whose times
were never aligned.

> **Status.** Tested on one 28.6-minute folktale recording in Afade (Chadic),
> and on parts of it, on a laptop CPU (Windows) and on a Linux cluster GPU.
> The boundaries it produces have **not been checked by ear**; see
> [Limitations](#limitations).

## Contents

- [How the two parts fit together](#how-the-two-parts-fit-together)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Part 1: `asr_to_eaf.py`](#part-1-asr_to_eafpy)
- [Part 2: `attach_reference.py`](#part-2-attach_referencepy)
- [Alignment score and "weak words"](#alignment-score-and-weak-words)
- [Limitations](#limitations)
- [Speed](#speed)
- [Troubleshooting](#troubleshooting)
- [Licenses](#licenses)
- [Citation](#citation)

## How the two parts fit together

```
recording.wav ──► asr_to_eaf.py ──► recording_asr.eaf        tier "ASR": one annotation per speech segment
                                          │
transcription.eaf (or .txt) ──────────────┤
                                          ▼
                 attach_reference.py ──► recording_asr_ref.eaf   tier "ASR" + all tiers of the transcription,
                                                                 on real time
```

Each part also works alone: part 1 without any transcription, part 2 without
an ASR file (it then writes a new `.eaf` linked to the recording).

The parts do not share segments. Part 1 cuts at pauses; part 2 keeps the
sentences of *your* transcription and finds their own start and end times.
In the merged file you see both, one under the other, so you can compare the
draft with your text, or use the ASR segments for the parts of the recording
your transcription does not cover.

## Installation

Python 3.9 or newer (tested with 3.12). The scripts are plain files: clone the
repository and run them from it.

```
git clone https://github.com/bomolopuu/elan-asr-align
cd elan-asr-align
```

**Part 2 only** (any system; Windows, macOS, Linux; CPU or GPU):

```
pip install -r requirements-align.txt
```

**Part 1, or both parts** (Linux; `fairseq2`, the library under omniASR, also
has builds for macOS 14+ on Apple Silicon, untested here, but none for Windows
or Intel Macs — on Windows use WSL):

```
pip install -r requirements-asr.txt
```

Both files pin the versions that were tested (see the comments in them). Other
versions of `torch`/`torchaudio` from 2.1 on should work for part 2:
`torchaudio.functional.forced_align` and the `MMS_FA` model were added in 2.1
and are still in 2.9.x and 2.10.0 (checked in the release files, not run).
torchaudio 2.8–2.9 print a deprecation warning for `forced_align`; it is
harmless, and the deprecation was dropped in 2.10. The pinned files were not
tested as a fresh install; they describe the environments in which the scripts
were run.

`soundfile` needs the system library `libsndfile` (included in the wheels on
most systems). For `.mp4`, `.m4a` and other formats that `libsndfile` cannot
read, `ffmpeg` must be on the `PATH`.

**Model weights** are downloaded on first use:

| what | size | where | override with |
|---|---|---|---|
| omniASR model (part 1) | e.g. 6.1 GB for `omniASR_LLM_300M_v2`, 30 GB for `omniASR_LLM_7B_ZS` | `~/.cache/fairseq2/assets` | `FAIRSEQ2_CACHE_DIR` |
| MMS_FA alignment model (part 2) | ~1.2 GB | `~/.cache/torch/hub` | `TORCH_HOME` |

On clusters with a small `$HOME` quota, point both variables to a larger disk.

## Quick start

### On one machine

```
# Part 1: segments + ASR draft (a GPU is used automatically if present)
python asr_to_eaf.py recording.wav --lang eng_Latn -o recording_asr.eaf

# Part 2: put your transcription onto real time and merge it in
python attach_reference.py recording.wav transcription.eaf --base recording_asr.eaf -o recording_asr_ref.eaf
```

Open `recording_asr_ref.eaf` in ELAN. The recording is linked by a relative
path, so keep the `.eaf` next to the audio (or in a folder the relative path
still points from).

### On a SLURM cluster

Copy [`slurm/asr_to_eaf.slurm`](slurm/asr_to_eaf.slurm), replace every `<…>`
(partition, environment, paths, language code) and submit it with `sbatch`.
The template runs part 1 and has part 2 commented out underneath.

## Part 1: `asr_to_eaf.py`

```
python asr_to_eaf.py AUDIO [-o OUT.eaf] [--model MODEL] [--lang CODE] [options]
```

What it does:

1. **Audio** → 16 kHz mono. From a stereo file the *louder* channel is taken,
   not the sum (in field recordings one channel is often dead or noisy).
2. **Speech segments** (`vad_core.py`, numpy only). The energy of the signal is
   compared with a noise floor that follows time (a fan, a change of distance to
   the microphone do not break it), using two thresholds that are fractions of
   the recording's own dynamic range, so that quiet and loud recordings work with
   the same settings. A pause of at least `--min-silence-ms` (default 500 ms)
   ends a segment; `--auto-segment` instead fits the pause length to the
   recording. Segments longer than `--max-segment-s` (15 s) are split at their
   longest pause.
3. **ASR**: every segment goes through the omniASR model, in batches
   (`--batch-size`). Pieces longer than 38 s are cut, because the models accept
   at most 40 s per input. Outputs that are one word or phrase repeated over and
   over (a typical hallucination on noise) are discarded.
4. **`.eaf`** with one time-aligned tier (`--tier`, default `ASR`). Segments for
   which the model returned nothing are left out unless `--keep-empty` is given
   — then the file also serves as a pure speech/pause segmentation.

### Models

`--model` takes the name of an omniASR model card (default `omniASR_LLM_300M_v2`):

| family | names | notes |
|---|---|---|
| CTC | `omniASR_CTC_{300M,1B,3B,7B}` and `…_v2` | fastest; `--lang` is ignored |
| LLM | `omniASR_LLM_{300M,1B,3B,7B}` and `…_v2` | better with `--lang` |
| LLM, long audio | `omniASR_LLM_Unlimited_{300M,1B,3B,7B}_v2` | not needed here: segments are short |
| zero-shot | `omniASR_LLM_7B_ZS` | for languages the model was not trained on; needs a few transcribed examples (below) |

Only `omniASR_LLM_300M_v2` was tested with this script. The 7B models need a
GPU with enough memory; on a CPU, use a 300M model.

### Language codes

LLM models take a language code of the form `ISO 639-3 + "_" + script`, e.g.
`eng_Latn`, `fra_Latn`, `hau_Latn`. The script refuses a code the model does
not know and suggests close ones. The full list:

```
python -c "from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs as s; print('\n'.join(s))"
```

If your language is not in the list, try the code of a closely related language
whose orthography is similar, a CTC model (no code at all), or the zero-shot
model.

### Zero-shot model with your own examples

`omniASR_LLM_7B_ZS` is conditioned on examples of the language: short
recordings (≤ 30 s) with their transcriptions, up to 10 (fewer are repeated).

```
python asr_to_eaf.py recording.wav --model omniASR_LLM_7B_ZS --context-tsv examples.tsv
```

`examples.tsv` has no header; each line is `audio path<TAB>transcription`
(relative paths are read from the folder of the TSV). Single pairs can also be
given as `--context-audio a.wav --context-text "…"`, repeated.

### Weights already on disk

Nodes without internet access, or a fine-tuned checkpoint: write a model card
(a small YAML file) that points to the local file, and give its name with the
suffix `@local`.

```yaml
# <PROJECT_DIR>/model_cards/omniASR_LLM_300M_v2.yaml
name: omniASR_LLM_300M_v2@local
checkpoint: file:///<MODELS_DIR>/omniASR-LLM-300M-v2.pt
```

```yaml
# <PROJECT_DIR>/model_cards/omniASR_tokenizer_written_v2.yaml
name: omniASR_tokenizer_written_v2@local
tokenizer_family: char_tokenizer
tokenizer: file:///<MODELS_DIR>/omniASR_tokenizer_written_v2.model
```

```
export FAIRSEQ2_ASSET_DIR=<PROJECT_DIR>/model_cards
python asr_to_eaf.py recording.wav --model omniASR_LLM_300M_v2@local --lang eng_Latn
```

The `@local` card inherits everything else from the official card. Put *only*
card files into that folder: fairseq2 reads every `.yaml` in it, and an
unrelated one (a training config, say) stops it with an `AssetMetadataError`.
The checkpoint and tokenizer URLs of the official cards are listed in
`omnilingual_asr/cards/models/*.yaml` inside the installed package.

### Options

| option | default | |
|---|---|---|
| `-o` | `<audio>_asr.eaf` | output file |
| `--model` | `omniASR_LLM_300M_v2` | model card |
| `--lang` | — | language code (LLM models) |
| `--batch-size` | 4 | segments per batch; lower it if the GPU runs out of memory |
| `--device` | `cuda` if available | `cpu` to force the CPU |
| `--min-silence-ms` | 500 | a pause at least this long ends a segment |
| `--auto-segment` | off | fit the pause length to the recording instead |
| `--min-speech-ms` | 200 | shorter sounds are ignored |
| `--max-segment-s` | 15 | longer segments are split at their longest pause |
| `--energy-threshold` | 0 (adaptive) | an absolute RMS level such as 0.05 makes the detector stricter on noisy recordings |
| `--pad-start-ms`, `--pad-end-ms` | 3, −5 | padding from the edge of the voice; negative = narrower |
| `--tier` | `ASR` | tier name |
| `--keep-empty` | off | keep segments with no ASR text |

## Part 2: `attach_reference.py`

```
python attach_reference.py AUDIO REFERENCE [--base ASR.eaf] [-o OUT.eaf] [options]
```

**Input.** `REFERENCE` is either

- an `.eaf` whose first top-level tier holds the sentences (or the tier given
  by `--ref-tier`), with any dependent tiers: words, morphemes, glosses, free
  translation, notes — symbolic or time-aligned; its times may be dummy
  (0–4 s, 5–9 s …) or missing; or
- a `.txt` file with one sentence per line.

The transcription may cover only part of the recording (the opening of a tale,
one episode …), and it must follow the order of the recording.

**What it does.**

1. **Emissions.** The multilingual alignment model
   [MMS_FA](https://pytorch.org/audio/stable/pipelines.html#torchaudio.pipelines.MMS_FA)
   is run over the recording (or the part given by `--search FROM:TO`, in
   seconds) in 30-second chunks.
2. **Forced alignment.** The text is romanised to the 26 Latin letters and the
   apostrophe that MMS_FA knows: tones and other diacritics are dropped,
   IPA and Africanist letters mapped (`ŋ→ng`, `ɛ→e`, `ɓ→b` …). It is aligned
   with a *star* token at both ends that absorbs any audio the transcription
   does not contain. `--star-between` also allows untranscribed audio between
   sentences (skipped repetitions, asides, the interviewer).
3. **Boundaries.** A sentence spans its first to its last aligned word. The
   boundary between two sentences is moved into the pause between them (energy
   track from `vad_core.py`), or to the energy minimum if there is no pause, but
   never into a word the alignment found. The outer edges are moved to the
   speech onset before the first word and the offset after the last.
4. **Merge.** Every tier of the transcription is copied with its hierarchy,
   linguistic types and constraints; annotation and time-slot IDs are
   renumbered so nothing clashes with the base file. Symbolic tiers (words,
   glosses, notes) follow their sentence automatically; time-aligned child
   tiers are stretched from the old sentence interval to the new one. If a tier
   of the transcription has the same name as a tier of the base file, the *base*
   tier is renamed (`--base-suffix`, default `_ASR`): your tier names are kept.

**Output.** The `.eaf`, a report of sentences with their times and alignment
score (below), and with `--words-tsv FILE` a table of word times.

**Example report** (made-up text):

```
Reference: tier 'Transcription', 3 sentences, 14 words, 42 letters
Audio: 742.6 s; searching 0.0–120.0 s on cpu

  # ref      start      end score  text
  1 1     00:03.12 00:05.40  0.71  tàmú bé kɔ̀lɔ́ ŋwà sì
  2 2     00:05.40 00:08.05  0.24  ɓèlé nà túmɔ̀ kà pɛ̀ dí  ← check
                                  weak words: túmɔ̀ kà
  3 3     00:08.61 00:10.20  0.66  sì tàmú lóŋ
Written: recording_asr_ref.eaf
```

| option | default | |
|---|---|---|
| `--base` | — | `.eaf` to merge into (e.g. from part 1); without it a new `.eaf` linked to the audio is written |
| `-o` | `<base or audio>_ref.eaf` | output file |
| `--ref-tier` | first top-level tier | tier with the sentences |
| `--search FROM:TO` | whole recording | look only in this part, seconds (`0:120`); much faster |
| `--star-between` | off | allow untranscribed audio between sentences |
| `--min-pause-ms` | 60 | shortest pause a boundary may be moved into |
| `--warn-score` | 0.30 | sentences below this score are marked `← check` |
| `--words-tsv` | — | also write word times |
| `--base-suffix` | `_ASR` | suffix for renamed base tiers |
| `--device` | `cuda` if available | |

### Why part 1 has no `--reference-text`

An earlier, private version of part 1 could spread a ready-made plain text over
its own pause-based segments (by forced alignment, or by a word-by-word sequence
alignment as a fallback). That is not included: the sequence alignment gave poor
results, and part 2 does the job better — it keeps the sentences of the
transcription instead of forcing them into pause-based segments, it allows the
text to cover only part of the recording, it works on recordings of any length,
and it carries over the dependent tiers. If you have a transcription, run part 1
for the draft and segmentation (or skip it), then part 2.

## Alignment score and "weak words"

The **score** of a sentence is the mean probability that the alignment model
gives to its letters at the places where it put them: 0–1, higher is better.
It is not a probability that the boundaries are right; it says how well the
sound matches the romanised letters. A sentence below `--warn-score` (0.30) is
marked `← check`.

**Weak words** are the words of a sentence whose own score is below 0.20. In
the test recording they marked places where the speaker said something other
than the written text, or said it very quietly; a typo in the transcription has
the same effect. They are where to listen first. A sentence can have a good
overall score and still contain a weak word.

The thresholds were set on one recording; treat them as a hint, not a rule.

## Limitations

- **The boundaries have not been checked by ear.** They were compared with an
  earlier, semi-manual alignment of the same recording (agreement within about
  0.3 s) and with the model's own free decoding, not with a listener.
- **Romanisation loses tones and diacritics.** The alignment model knows only
  `a–z` and `'`. Tone, nasalisation, vowel quality (`ɛ`/`e`, `ɔ`/`o`) and
  length do not exist for it, so words that differ only in these look the same,
  and a long text with many repeated words can be aligned to the wrong
  repetition. The original spelling is kept in the output; only the alignment
  ignores it.
- **The star tokens can narrow the edges.** The star absorbs audio outside the
  transcription, and near the first and last sentence it may also take a quiet
  onset or a fading end of a word. Check the start of the first sentence and the
  end of the last.
- **The order matters.** The transcription must follow the recording.
  Sentences that are missing from the text are absorbed only at the edges (or
  between sentences with `--star-between`); sentences in the text that are not
  in the recording, or are in a different order, are forced onto the wrong audio
  (look for low scores).
- **omniASR does not know your language** unless it was in its training data,
  and even then accuracy for field recordings is not guaranteed. Treat the
  ASR tier as a draft to correct, not a transcription.
- **Speech detection is energy-based.** Loud noise, music or a second speaker
  count as speech; overlapping speakers end up in one segment and one tier.
  Speaker diarization is not included: the tools for it need heavy additional
  dependencies (and crashed in the same process as fairseq2 in our tests), and
  automatic speaker counting was not reliable on field recordings.
- **CPU is slow** (see [Speed](#speed)). Use `--search` in part 2 when you know
  roughly where the transcribed part is.

## Speed

Measured on the 28.6-minute test recording and on its first 90 seconds:

| task | machine | time |
|---|---|---|
| part 2, whole recording (28.6 min), transcription of 4 sentences | laptop CPU (16 threads) | 22 min |
| part 2, `--search 0:60` | laptop CPU | 40 s – 2 min |
| part 2, whole recording | GPU (NVIDIA H100 NVL, shared) | 17 s |
| part 2, 90 s of audio | GPU (NVIDIA H100 NVL, shared) | 5 s |
| part 1, 90 s of audio (24 segments), `omniASR_LLM_300M_v2` | GPU (NVIDIA H100 NVL, shared) | 28 s |

Times include loading the models; the weights were already downloaded.
Part 1 also runs on a CPU (checked on a cluster node, not timed), several
times slower than on a GPU (the first batches took about ten times as long).

## Troubleshooting

| message | what to do |
|---|---|
| `forced_align has been deprecated …` | harmless warning of torchaudio 2.8–2.9 |
| `libffi.so.7` / `libssl.so.1.1: cannot open shared object file` | `fairseq2n` was built against older system libraries; ask the cluster admins for them, or put copies into a folder and add it to `LD_LIBRARY_PATH` |
| `AssetMetadataError` | `FAIRSEQ2_ASSET_DIR` contains `.yaml` files that are not model cards |
| `CUDA out of memory` | lower `--batch-size`, or use a smaller model |
| `--lang 'xxx' is not among the … language codes` | a typo or a language the model does not have; see [Language codes](#language-codes) |
| `The reference (N letters) is too long for … s of audio` | `--search` is too narrow for the transcription |
| `Cannot read … install ffmpeg` | the format needs `ffmpeg`, or convert the file to `.wav` first |

## Licenses

- **This code**: © 2026 Tatiana Korol,
  [Creative Commons Attribution-NonCommercial 4.0](LICENSE) (CC BY-NC 4.0).
  You may use, share and adapt it for **non-commercial** purposes only, and you
  must **credit the author**, e.g.:

  > Tatiana Korol, *elan-asr-align* (2026),
  > https://github.com/bomolopuu/elan-asr-align, licensed under CC BY-NC 4.0.

  In a publication, cite it as under [Citation](#citation). For commercial
  use, ask the author for permission.
- **MMS_FA weights** (downloaded by `torchaudio`, used by part 2): CC BY-NC 4.0,
  per the `torchaudio` documentation of the model.
- **omniASR** (`omnilingual-asr`): code and model weights under Apache 2.0, per
  its repository.
- **torchaudio**: BSD 2-Clause; **fairseq2**: MIT.

The licenses of the models apply to your use of them, independently of the
license of this code.

## Citation

If you use these tools in research, please cite this repository (see
[`CITATION.cff`](CITATION.cff); GitHub shows a "Cite this repository" button)
and the models it relies on:

- Omnilingual ASR team (2025). *Omnilingual ASR: Open-Source Multilingual Speech
  Recognition for 1600+ Languages.* arXiv:2511.09690.
- Pratap, V. et al. (2024). *Scaling Speech Technology to 1,000+ Languages.*
  Journal of Machine Learning Research 25(97). (MMS, the source of MMS_FA.)
