#!/usr/bin/env python3
"""Dub an English video into German, fully local. Version 2: a cloned voice per speaker.

Pipeline: ffmpeg -> Demucs (split voice/background) -> Whisper (words) + pyannote (who speaks when)
-> lines (+ voiceprint check) -> Ollama LLM (translate) -> Qwen3-TTS or XTTS v2 (each speaker's
cloned voice) -> ffmpeg. Every step caches its result in the work dir, so you can edit and re-run.

Adds to dub_v1.py: lines cut at sentence ends and speaker changes; every line's speaker checked
against the speakers' voiceprints (fixes quick back-and-forth exchanges), speakers that sound like
one person merged; a cloned voice per speaker, and a translation that knows who speaks (role,
gender); MKV and other files with several audio tracks or subtitles: the English track is dubbed,
all tracks are kept. Needs pyannote (uv sync --extra speakers) and a Hugging Face login.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # XTTS v2: non-commercial CPML license
# AMD: by default MIOpen benchmarks kernels for every new conv shape, and every XTTS clip has
# a new length. FAST skips that (~5x faster XTTS). Level 3 hides its "IsEnoughWorkspace" warnings.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm GPUs show up as "cuda"
SR = 24000  # output sample rate of both TTS engines
QWEN3_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # the "Base" variant is the one that clones voices

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"  # gated: accept its terms on Hugging Face
# Voiceprint similarity (cosine) between two "speakers" that are one person measured 0.55-0.68,
# between different people at most 0.29.
SAME_PERSON = 0.45
# A line moves to another speaker if its voice matches them at least this well, and by this much
# more than the speaker it has.
MIN_MATCH, MIN_GAIN = 0.25, 0.15
TEXT_SUBTITLES = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}  # the kinds MP4 can hold

CONTEXT_PROMPT = """Below is the English transcript of a video that will be dubbed into German.
Each line starts with the letter of its speaker.
Answer with a JSON object:
- "summary": 2-3 sentences: what the video is about, who it is for, its tone.
- "speakers": an object mapping each speaker letter to who it is (name and role, for example
  "the host" or "an interviewed scientist"), as far as the transcript tells; don't invent names.
- "glossary": an object mapping every name (people, shows, places, organisations, missions,
  spacecraft, products) and technical term to how it should appear in German.
  Keep names unchanged; use the usual German term for everything else.

TRANSCRIPT:
{transcript}"""

PROMPT = """You translate English video dialogue into German for a voice-over dub.
About the video: {summary}
Glossary, use exactly these: {glossary}

Translate only the LINE, spoken by {speaker}. Rules:
- Natural spoken German, same meaning and tone. Address the audience with "{address}".
- It must be spoken in about {seconds:.1f} seconds, so keep it as short as the English; rephrase rather than add words.
- Keep every fact, name and number exactly. Write numbers as digits, like the English.
- Answer with the German line only: no quotes, no notes, no English.

Previous lines (speaker: English -> German):
{before}
Following lines (speaker: English, only for context): {after}

LINE: {line}"""

CHECK_PROMPT = """You review one line of a German voice-over dub against its English original.
About the video: {summary}
Glossary: {glossary}

Spoken by {speaker}.
English: {en}
German: {de}
{hint}
Check the German for: changed or missing facts, wrong numbers or names, misspelled or broken words,
leftover English (names and glossary terms are fine), wrong form of address (must be "{address}"),
forms that don't fit the speaker (a woman says "ich bin Ingenieurin", not "Ingenieur").
Only fix real errors; don't rephrase a correct line.
If the German is correct, answer exactly OK. Otherwise answer with the corrected German line only."""


def step(msg):
    print(f"\n=== {msg}", flush=True)


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def free_gpu():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def streams(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_entries",
                        "stream=codec_type,codec_name:stream_tags=language", str(video)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"]


def english_track(video, track):
    """Which audio track to dub (counted from 0): the given one, else the first tagged English,
    else the first. ffmpeg on its own would take the one with the most channels."""
    audio = [s.get("tags", {}).get("language", "?") for s in streams(video) if s["codec_type"] == "audio"]
    if not audio:
        raise SystemExit(f"{video} has no audio track")
    if track is None:
        track = next((n for n, lang in enumerate(audio) if lang.lower() in ("eng", "en")), 0)
    if len(audio) > 1:
        print(f"Audio tracks: {', '.join(f'{n}={lang}' for n, lang in enumerate(audio))} -> dubbing track {track}")
    return track


def extract_audio(video, out, track):
    run("ffmpeg", "-y", "-loglevel", "error", "-i", video, "-map", f"0:a:{track}", "-ac", "2", "-ar", "44100", out)


def separate(audio, vocals_out, bg_out):
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs").to(DEV).eval()
    wav, sr = sf.read(audio, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.T.copy())
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        src = apply_model(model, ((wav - mean) / std)[None], device=DEV,
                          split=True, overlap=0.25, progress=True)[0]
    src = src * std + mean
    vocals = src[model.sources.index("vocals")]
    sf.write(vocals_out, vocals.T.numpy(), sr)
    sf.write(bg_out, (src.sum(0) - vocals).T.numpy(), sr)
    del model, src
    free_gpu()


def transcribe(vocals, model_name):
    """Every spoken word with its start and end time."""
    import whisper

    model = whisper.load_model(model_name, device=DEV)
    res = model.transcribe(whisper.load_audio(str(vocals)), language="en", fp16=DEV == "cuda",
                           condition_on_previous_text=False, word_timestamps=True)
    del model
    free_gpu()
    return [{"start": w["start"], "end": w["end"], "word": w["word"].strip()}
            for s in res["segments"] for w in s.get("words", []) if w["word"].strip()]


def diarize(vocals, speakers):
    """Who speaks when (pyannote): [start, end, speaker] turns, one speaker at a time."""
    import librosa
    from pyannote.audio import Pipeline

    wav, sr = sf.read(vocals, dtype="float32", always_2d=True)
    wav = torch.from_numpy(librosa.resample(wav.mean(1), orig_sr=sr, target_sr=16000))[None]
    try:
        pipe = Pipeline.from_pretrained(DIARIZATION_MODEL).to(torch.device(DEV))
    except Exception as e:
        raise SystemExit(f"Can't load {DIARIZATION_MODEL} ({e}).\nIt's gated: accept its terms on "
                         f"https://huggingface.co/{DIARIZATION_MODEL}, then `uv run hf auth login`. "
                         "Or use --speakers 1 for one voice without diarization.")
    with torch.backends.cudnn.flags(enabled=False):  # MIOpen's LSTM fails on RDNA 3 (miopenStatusUnknownError)
        out = pipe({"waveform": wav, "sample_rate": 16000}, num_speakers=speakers)
    del pipe
    free_gpu()
    return [[t.start, t.end, spk] for t, spk in out.exclusive_speaker_diarization]


ABBREVIATIONS = ("Mr.", "Mrs.", "Ms.", "Dr.", "St.", "vs.", "U.S.", "e.g.", "i.e.", "etc.")


def make_lines(words, turns, max_len=15.0, min_turn=1.5):
    """Cut the words into lines at sentence ends and speaker changes, so every line is one speaker
    saying whole sentences. A sentence over max_len is split at its longest pause."""
    sentences, cur = [], []
    for w in words:
        w["speaker"] = speaker_at(turns, w["start"], w["end"])
        cur.append(w)
        if w["word"].endswith((".", "?", "!")) and not w["word"].endswith(ABBREVIATIONS):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    def runs(ws):  # consecutive words of the same speaker
        out = []
        for w in ws:
            if out and out[-1][-1]["speaker"] == w["speaker"]:
                out[-1].append(w)
            else:
                out.append([w])
        return out

    def length(ws):
        return ws[-1]["end"] - ws[0]["start"]

    lines = []
    for ws in sentences:
        # A speaker change inside a sentence that lasts under min_turn is nearly always the
        # diarization being a word or two off at a turn boundary: those words go to the neighbour.
        while len(r := runs(ws)) > 1 and length(short := min(r, key=length)) < min_turn:
            k = r.index(short)
            neighbour = max((r[j] for j in (k - 1, k + 1) if 0 <= j < len(r)), key=length)
            for w in short:
                w["speaker"] = neighbour[0]["speaker"]
        lines += runs(ws)

    def split(ws):  # at the longest pause, and rather after a comma
        if ws[-1]["end"] - ws[0]["start"] <= max_len or len(ws) < 2:
            return [ws]
        k = max(range(1, len(ws)), key=lambda j: ws[j]["start"] - ws[j - 1]["end"]
                + (0.5 if ws[j - 1]["word"].endswith((",", ";", ":")) else 0))
        return split(ws[:k]) + split(ws[k:])

    return [{"start": ws[0]["start"], "end": ws[-1]["end"], "speaker": ws[0]["speaker"],
             "en": " ".join(w["word"] for w in ws)} for line in lines for ws in split(line)]


def recheck_speakers(lines, vocals, merge):
    """Second opinion on who speaks each line, from voiceprints: diarization blurs quick exchanges,
    but a line's own voice still matches its speaker. Speakers that sound like one person are
    merged first (unless merge is False). Returns what changed, for the log."""
    import librosa
    from pyannote.audio import Pipeline

    wav, sr = sf.read(vocals, dtype="float32", always_2d=True)
    wav = librosa.resample(wav.mean(1), orig_sr=sr, target_sr=16000)
    pipe = Pipeline.from_pretrained(DIARIZATION_MODEL).to(torch.device(DEV))
    voiceprint_model = pipe._embedding  # the model pyannote groups speakers with (no public accessor)
    emb = []
    with torch.backends.cudnn.flags(enabled=False):  # see diarize()
        for s in lines:
            x = wav[int(s["start"] * 16000):int(s["end"] * 16000)]
            x = np.pad(x, (0, max(0, 1600 - len(x))))  # at least 0.1 s
            e = np.nan_to_num(voiceprint_model(torch.from_numpy(x.copy())[None, None])[0])
            emb.append(e / (np.linalg.norm(e) + 1e-9))
    del pipe, voiceprint_model
    free_gpu()
    emb = np.array(emb)
    dur = np.array([s["end"] - s["start"] for s in lines])

    def voiceprint(spk, skip=None):  # duration-weighted, from lines of 2 s or more where possible
        mine = [i for i, s in enumerate(lines) if s["speaker"] == spk and i != skip]
        mine = [i for i in mine if dur[i] >= 2.0] or mine
        if not mine:
            return None
        v = (emb[mine] * dur[mine, None]).sum(0)
        return v / (np.linalg.norm(v) + 1e-9)

    changes = []
    while merge:
        spks = sorted({s["speaker"] for s in lines})
        vp = {k: voiceprint(k) for k in spks}
        pairs = [(float(vp[a] @ vp[b]), a, b) for n, a in enumerate(spks) for b in spks[n + 1:]]
        if not pairs or max(pairs)[0] < SAME_PERSON:
            break
        sim, a, b = max(pairs)
        talk = {k: sum(d for d, s in zip(dur, lines) if s["speaker"] == k) for k in (a, b)}
        keep, drop = (a, b) if talk[a] >= talk[b] else (b, a)
        for s in lines:
            if s["speaker"] == drop:
                s["speaker"] = keep
        changes.append(f"two speakers sound like one person (voiceprints {sim:.2f} alike): merged")

    def sims(i):  # how well line i's voice matches each speaker, leaving the line itself out
        return {k: float(emb[i] @ v) for k in {s["speaker"] for s in lines} if (v := voiceprint(k, skip=i)) is not None}

    for _ in range(3):  # voiceprints get cleaner with every fix, so look again
        moved = False
        for i, s in enumerate(lines):
            m = sims(i)
            best, now = max(m, key=m.get), m.get(s["speaker"], -1.0)
            if best != s["speaker"] and m[best] >= MIN_MATCH and m[best] - now >= MIN_GAIN:
                changes.append(f'{s["start"]:7.1f}s  voice {m[best]:.2f} vs {now:.2f}: moved to another speaker: {s["en"][:60]}')
                s["speaker"], moved = best, True
        if not moved:
            break

    # A short reply the voice can't place: a question is rarely answered by whoever asked it.
    for i in range(1, len(lines)):
        s, q = lines[i], lines[i - 1]
        if (dur[i] < 1.0 and q["en"].endswith("?") and q["speaker"] == s["speaker"]
                and s["start"] - q["end"] < 2.0 and sims(i).get(s["speaker"], 0.0) < MIN_MATCH):
            other = next((x["speaker"] for x in reversed(lines[max(0, i - 5):i - 1]) if x["speaker"] != s["speaker"]),
                         lines[i + 1]["speaker"] if i + 1 < len(lines) and lines[i + 1]["speaker"] != s["speaker"] else None)
            if other:
                changes.append(f'{s["start"]:7.1f}s  reply to a question: moved to another speaker: {s["en"][:60]}')
                s["speaker"] = other
    return changes


def finish_lines(lines, max_gap=0.3, merge_len=8.0):
    """Join quick sentences of one speaker into one line, and name the speakers A, B, ... by how
    much they talk."""
    segs = []
    for s in lines:
        s = dict(s)
        p = segs[-1] if segs else None
        # quick sentences in one breath ("I don't know. That's what we do.") stay together
        if (p and p["speaker"] == s["speaker"] and s["start"] - p["end"] <= max_gap
                and s["end"] - p["start"] <= merge_len):
            p["end"], p["en"] = s["end"], f'{p["en"]} {s["en"]}'
        else:
            segs.append(s)
    # speakers as letters, A = who talks most
    talk = {}
    for s in segs:
        talk[s["speaker"]] = talk.get(s["speaker"], 0) + s["end"] - s["start"]
    letters = {spk: chr(65 + n) for n, spk in enumerate(sorted(talk, key=talk.get, reverse=True))}
    for s in segs:
        s["speaker"] = letters[s["speaker"]]
    return segs


def speaker_at(turns, start, end):
    """The speaker who talks most during [start, end], else the one closest in time."""
    if not turns:
        return "A"
    best = max(turns, key=lambda t: min(t[1], end) - max(t[0], start))
    if min(best[1], end) - max(best[0], start) > 0:
        return best[2]
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]


def ollama(url, model, prompt, keep_alive="10m", fmt=None, num_ctx=None):
    # think=False: reasoning models (gemma4, qwen3) would otherwise think before every line, ~20x slower
    body = {"model": model, "stream": False, "think": False, "keep_alive": keep_alive,
            "options": {"temperature": 0.2}}
    if prompt is not None:
        body["prompt"] = prompt
    if fmt:
        body["format"] = fmt
    if num_ctx:
        body["options"]["num_ctx"] = num_ctx
    for penalty in (None, 1.1):  # retry once if the model loops
        if penalty:
            body["options"]["repeat_penalty"] = penalty
        req = urllib.request.Request(f"{url}/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r).get("response", "")
        except urllib.error.HTTPError as e:
            error = e.read().decode(errors="replace")
            if penalty or "repeat limit" not in error:
                raise SystemExit(f"Ollama failed: {error}") from e


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return " ".join(text.split()).strip('"„“”\' ')


def analyze(segs, a):
    """Read the whole transcript once: a summary, who the speakers are, and a glossary of names
    and terms for every line."""
    transcript = "\n".join(f'{s["speaker"]}: {s["en"]}' for s in segs)
    # the default context window (4096 tokens) is too small for a whole episode; ~3 chars per token
    num_ctx = (len(CONTEXT_PROMPT + transcript) // 3 // 4096 + 2) * 4096
    try:
        ctx = json.loads(ollama(a.ollama_url, a.llm, CONTEXT_PROMPT.format(transcript=transcript),
                                fmt="json", num_ctx=num_ctx))
    except json.JSONDecodeError:
        ctx = {}
    return {"summary": str(ctx.get("summary", "")), "speakers": dict(ctx.get("speakers") or {}),
            "glossary": dict(ctx.get("glossary") or {})}


def who(ctx, voices, spk):
    """'B (the host; a woman)': what the translator knows about a speaker."""
    role, gender = ctx.get("speakers", {}).get(spk), voices.get(spk, {}).get("gender")
    facts = ([str(role)] if role else []) + ([f"a {gender}"] if gender else [])
    return f"{spk} ({'; '.join(facts)})" if facts else spk


def glossary_for(ctx, text):
    """Only the glossary entries that occur in this text: keeps the prompts short."""
    hits = [f"{en} = {de}" for en, de in ctx["glossary"].items() if en.lower() in text.lower()]
    return "; ".join(hits) or "-"


def numbers(text):
    return sorted(re.sub(r"[.,]", "", n) for n in re.findall(r"\d[\d.,]*\d|\d", text))


def translate_line(segs, i, ctx, voices, a):
    s = segs[i]
    before = "\n".join(f'{x["speaker"]}: {x["en"]} -> {x.get("de", "?")}' for x in segs[max(0, i - 3):i]) or "-"
    after = " ".join(f'{x["speaker"]}: {x["en"]}' for x in segs[i + 1:i + 3]) or "-"
    return clean(ollama(a.ollama_url, a.llm, PROMPT.format(
        summary=ctx["summary"] or "-", glossary=glossary_for(ctx, f"{before} {s['en']} {after}"),
        speaker=who(ctx, voices, s["speaker"]), address=a.address, seconds=s["end"] - s["start"],
        line=s["en"], before=before, after=after)))


def check_line(s, ctx, voices, a):
    """Second opinion on one translated line. Returns the corrected line, or None if it's fine."""
    # only numbers the German lost: a German "4" for an English "four" is fine
    missing = [n for n in numbers(s["en"]) if n not in numbers(s["de"])]
    hint = f"Note: the German is missing the number(s) {', '.join(missing)} from the English.\n" if missing else ""
    answer = clean(ollama(a.ollama_url, a.llm, CHECK_PROMPT.format(
        summary=ctx["summary"] or "-", glossary=glossary_for(ctx, s["en"]), en=s["en"], de=s["de"],
        speaker=who(ctx, voices, s["speaker"]), hint=hint, address=a.address)))
    if re.sub(r"\W", "", answer).upper() == "OK" or answer == s["de"]:
        return None
    # A real fix never drops a third of the line; that's the reviewer cutting content (it does so
    # on lines that start mid-sentence)
    return answer if len(answer) >= 0.7 * len(s["de"]) else None


def translate(segs, path, ctx, voices, a):
    for i, s in enumerate(segs):
        if "de" not in s:
            s["de"] = translate_line(segs, i, ctx, voices, a)
            print(f'[{i + 1}/{len(segs)}] {s["speaker"]}: {s["en"]}\n             -> {s["de"]}')
        if s["de"] and not s.get("checked"):
            fixed = check_line(s, ctx, voices, a)
            if fixed:
                print(f'  check fixed line {i}: {s["de"]}\n                   -> {fixed}')
                s["de_unchecked"], s["de"] = s["de"], fixed
            s["checked"] = True
        path.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
    ollama(a.ollama_url, a.llm, None, keep_alive=0)  # unload the LLM to free VRAM for the TTS model


def make_voices(vocals, segs, voice_dir, seconds=15):
    """A voice sample per speaker (<letter>.wav + its transcript .txt) from their longest lines,
    whole lines only, plus voices.json with how much they speak and a guess at their gender."""
    import librosa

    voice_dir.mkdir(exist_ok=True)
    wav, sr = sf.read(vocals, dtype="float32", always_2d=True)
    mono, info = wav.mean(1), {}
    for spk in sorted({s["speaker"] for s in segs}):
        mine = sorted((s for s in segs if s["speaker"] == spk), key=lambda s: s["end"] - s["start"], reverse=True)
        picked, total = [], 0.0
        for s in mine:
            picked.append(s)
            total += s["end"] - s["start"]
            if total >= seconds:
                break
        picked.sort(key=lambda s: s["start"])
        sample = np.concatenate([mono[int(s["start"] * sr):int(s["end"] * sr)] for s in picked])
        sf.write(voice_dir / f"{spk}.wav", sample, sr)
        (voice_dir / f"{spk}.txt").write_text(" ".join(s["en"] for s in picked))
        # median pitch: men mostly speak at 85-155 Hz, women at 165-255 Hz
        f0, voiced, _ = librosa.pyin(librosa.resample(sample[:sr * 20], orig_sr=sr, target_sr=16000),
                                     fmin=65, fmax=400, sr=16000)
        pitch = float(np.nanmedian(f0[voiced])) if voiced.any() else 0.0
        info[spk] = {"seconds": round(sum(s["end"] - s["start"] for s in mine), 1), "pitch_hz": round(pitch),
                     "gender": "woman" if pitch > 160 else "man" if pitch else ""}
    (voice_dir / "voices.json").write_text(json.dumps(info, indent=1))


def clip_path(tts_dir, i, speaker, text):
    # speaker and text in the name: an edited line, or a line given to another speaker, is re-voiced
    return tts_dir / f"{i:04d}_{hashlib.md5(f'{speaker}|{text}'.encode()).hexdigest()[:8]}.wav"


def synthesize(segs, voices, tts_dir, engine, batch):
    """voices: speaker -> (sample wav, its transcript or None for timbre-only cloning)"""
    tts_dir.mkdir(exist_ok=True)
    todo = [(i, s["speaker"], s["de"]) for i, s in enumerate(segs)
            if s["de"] and not clip_path(tts_dir, i, s["speaker"], s["de"]).exists()]
    if not todo:
        return
    if engine == "qwen3":
        synth_qwen3(todo, voices, tts_dir, batch)
    else:
        synth_xtts(todo, voices, tts_dir)
    free_gpu()


def synth_xtts(todo, voices, tts_dir):
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEV)
    for n, (i, spk, text) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {spk}: {text}")
        wav = tts.tts(text=text, speaker_wav=str(voices[spk][0]), language="de")
        sf.write(clip_path(tts_dir, i, spk, text), np.asarray(wav, dtype=np.float32), SR)


def synth_qwen3(todo, voices, tts_dir, batch):
    import librosa
    from qwen_tts import Qwen3TTSModel
    from transformers.utils import logging

    logging.set_verbosity_error()  # hides "Setting `pad_token_id` to `eos_token_id`" on every batch
    tts = Qwen3TTSModel.from_pretrained(QWEN3_MODEL, device_map=f"{DEV}:0" if DEV == "cuda" else DEV,
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    # One prompt per speaker, built once. Without a transcript only the voice's timbre is cloned, so
    # the German comes out native; with it the model continues the English sample and keeps its accent.
    prompts = {spk: tts.create_voice_clone_prompt(ref_audio=str(voices[spk][0]), ref_text=voices[spk][1],
                                                  x_vector_only_mode=voices[spk][1] is None)[0]
               for spk in {spk for _, spk, _ in todo}}
    todo = sorted(todo, key=lambda t: len(t[2]))  # similar lengths in a batch: less padding
    n = 0
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        texts = [text for _, _, text in chunk]
        # The batch runs until its longest line ends, so cap the length (~12.5 codec frames per
        # second, generously ~6 characters per second): one runaway line can't stall the batch.
        wavs, sr = tts.generate_voice_clone(text=texts, language=["german"] * len(chunk),
                                            voice_clone_prompt=[prompts[spk] for _, spk, _ in chunk],
                                            max_new_tokens=int(12.5 * (max(map(len, texts)) / 6 + 5)))
        for (i, spk, text), wav in zip(chunk, wavs):
            n += 1
            print(f"[{n}/{len(todo)}] {spk}: {text}")
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            sf.write(clip_path(tts_dir, i, spk, text), np.asarray(wav, dtype=np.float32), SR)


def atempo(clip, factor):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                        "-i", "pipe:0", "-filter:a", f"atempo={factor:.3f}", "-f", "f32le", "pipe:1"],
                       input=clip.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)


def assemble(segs, tts_dir, out, total, max_speedup):
    """Place every German clip at its original timestamp; speed it up if it doesn't fit."""
    import librosa

    track = np.zeros(int(total * SR), np.float32)
    cursor, late = 0.0, []
    for i, s in enumerate(segs):
        if not s["de"]:
            continue
        clip, _ = sf.read(clip_path(tts_dir, i, s["speaker"], s["de"]), dtype="float32")
        clip, _ = librosa.effects.trim(clip, top_db=35)
        start = max(s["start"], cursor)
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else total
        slot = max(nxt - start, 0.2)
        if len(clip) / SR > slot:
            clip = atempo(clip, min(len(clip) / SR / slot, max_speedup))
        a = int(start * SR)
        if a + len(clip) > len(track):
            track = np.pad(track, (0, a + len(clip) - len(track)))
        track[a:a + len(clip)] += clip
        cursor = start + len(clip) / SR
        if cursor > nxt + 0.5:
            late.append((i, cursor - nxt))
    sf.write(out, track[:int(total * SR)], SR)
    return late


def mix(bg, voice, out, bg_volume):
    run("ffmpeg", "-y", "-loglevel", "error", "-i", bg, "-i", voice, "-filter_complex",
        f"[0:a]volume={bg_volume}[b];[1:a]aresample=44100,pan=stereo|c0=c0|c1=c0[v];"
        "[b][v]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[m]",
        "-map", "[m]", out)


def mux(video, audio, out):
    """German track first and default; all original audio tracks and subtitles kept as they are
    (and fonts in MKV). MP4 can only hold text subtitles, converted to its own format."""
    subs = [s["codec_name"] for s in streams(video) if s["codec_type"] == "subtitle"]
    mkv = out.suffix.lower() == ".mkv"
    keep_subs = bool(subs) and (mkv or all(c in TEXT_SUBTITLES for c in subs))
    if subs and not keep_subs:
        print(f"  ! {out.suffix} can't hold image subtitles ({', '.join(subs)}): use -o <name>.mkv to keep them")
    run("ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
        "-map", "0:V:0", "-map", "1:a:0", "-map", "0:a", *(["-map", "0:s"] if keep_subs else []),
        *(["-map", "0:t?"] if mkv else []),
        "-c", "copy", "-c:a:0", "aac", "-b:a:0", "192k", *(["-c:s", "mov_text"] if keep_subs and not mkv else []),
        "-metadata:s:a:0", "language=deu", "-metadata:s:a:0", "title=Deutsch (KI)",
        "-disposition:a", "0", "-disposition:a:0", "default", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="default: <video>_de.<same type> (.mkv for unusual types)")
    ap.add_argument("--audio-track", type=int, help="audio track to dub, counted from 0 (default: the English one)")
    ap.add_argument("--work", type=Path, default=Path("work"), help="cache dir (default: ./work)")
    ap.add_argument("--whisper", default="turbo", help="turbo (fast, default) or large-v3 (most accurate)")
    ap.add_argument("--llm", default="gemma4:12b", help="Ollama model for translation")
    ap.add_argument("--address", choices=["du", "Sie"], default="du", help="how to address the audience")
    ap.add_argument("--speakers", type=int,
                    help="number of speakers, if you know it (default: detected); 1 = one voice for all")
    ap.add_argument("--tts", choices=["qwen3", "xtts"], default="qwen3", help="TTS engine (default: qwen3)")
    ap.add_argument("--batch", type=int, default=8, help="lines per Qwen3-TTS batch (more = faster, more VRAM)")
    ap.add_argument("--ref", type=Path, help="one voice sample (10-30 s WAV) for all speakers; default: "
                                             "each speaker's own voice")
    ap.add_argument("--clone", choices=["timbre", "full"], default="timbre",
                    help="Qwen3-TTS: timbre = voice only, native German (default); "
                         "full = also imitates the English sample: closer voice, but English accent")
    ap.add_argument("--ref-text", help="what is said in --ref (only for --clone full); default: transcribed")
    ap.add_argument("--max-speedup", type=float, default=1.3, help="max speed-up for long German lines")
    ap.add_argument("--bg-volume", type=float, default=1.0, help="music/SFX volume in the mix")
    ap.add_argument("--stop-after", choices=["separate", "transcribe", "translate", "tts"])
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    a = ap.parse_args()

    if a.speakers != 1 and importlib.util.find_spec("pyannote") is None:  # say so before an hour of work
        raise SystemExit("Speaker detection needs pyannote: uv sync --extra speakers\n"
                         "Or use --speakers 1 for one voice without speaker detection.")
    # All of v2's files, even the audio: v1 may have dubbed another track of the same file
    w = a.work / a.video.stem / "v2"
    w.mkdir(parents=True, exist_ok=True)
    ext = a.video.suffix.lower() if a.video.suffix.lower() in (".mp4", ".mkv", ".mov", ".m4v") else ".mkv"
    out = a.out or a.video.with_name(f"{a.video.stem}_de{ext}")
    print(f"Device: {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'CPU (no GPU found!)'}")

    audio, vocals, bg = w / "audio.wav", w / "vocals.wav", w / "background.wav"
    if not audio.exists():
        step("Extracting audio")
        extract_audio(a.video, audio, english_track(a.video, a.audio_track))
    if not vocals.exists():
        step("Separating voice from music/SFX (Demucs)")
        separate(audio, vocals, bg)
    if a.stop_after == "separate":
        return

    seg_en, seg_de = w / "segments_en.json", w / "segments_de.json"
    if not seg_en.exists():
        words, dia = w / "words.json", w / "diarization.json"
        if not words.exists():
            step(f"Transcribing (Whisper {a.whisper}, with word times)")
            words.write_text(json.dumps(transcribe(vocals, a.whisper), indent=0))
        turns = []
        if a.speakers != 1:
            if not dia.exists():
                step("Finding who speaks when (pyannote)")
                dia.write_text(json.dumps(diarize(vocals, a.speakers), indent=0))
            turns = json.loads(dia.read_text())
        lines = make_lines(json.loads(words.read_text()), turns)
        if turns:
            step("Checking every line's voice against the speakers' voiceprints")
            changes = recheck_speakers(lines, vocals, merge=a.speakers is None)  # --speakers N: keep N
            print("\n".join(f"  {c}" for c in changes) or "  no changes")
        seg_en.write_text(json.dumps(finish_lines(lines), indent=1))
    if a.stop_after == "transcribe":
        return

    # segments_de.json wins once it exists, so your manual edits (also of speaker letters) are kept
    segs = json.loads((seg_de if seg_de.exists() else seg_en).read_text())
    voice_dir = w / "voices"
    if not (voice_dir / "voices.json").exists():
        step("Voice sample per speaker")
        make_voices(vocals, segs, voice_dir)
    voices = json.loads((voice_dir / "voices.json").read_text())  # edit "gender" here if it's wrong
    for spk, v in voices.items():
        print(f'  {spk}: {v["seconds"]:6.1f}s of speech, {v["pitch_hz"]} Hz -> {v["gender"] or "?"}')

    if any("de" not in s or (s["de"] and not s.get("checked")) for s in segs):
        ctx_path = w / "context.json"
        if not ctx_path.exists():  # like segments_de.json: edit it, and your version wins
            step(f"Reading the whole transcript ({a.llm})")
            ctx_path.write_text(json.dumps(analyze(segs, a), ensure_ascii=False, indent=1))
        ctx = json.loads(ctx_path.read_text())
        for spk, role in ctx.get("speakers", {}).items():
            print(f"  {spk}: {role}")
        step(f"Translating and checking ({a.llm})")
        translate(segs, seg_de, ctx, voices, a)
    if a.stop_after == "translate":
        print(f"\nReview/edit {seg_de}, then run again without --stop-after.")
        return

    full = a.tts == "qwen3" and a.clone == "full"  # the transcripts are only needed then
    if a.ref:
        ref_text = (a.ref_text or " ".join(w["word"] for w in transcribe(a.ref, a.whisper))) if full else None
        samples = {spk: (a.ref, ref_text) for spk in voices}
    else:
        samples = {spk: (voice_dir / f"{spk}.wav", (voice_dir / f"{spk}.txt").read_text() if full else None)
                   for spk in voices}
    for s in segs:  # a speaker letter you typed that has no voice sample gets A's voice
        if s["speaker"] not in samples:
            print(f'  ! line has speaker {s["speaker"]}, who has no voice sample: using A')
            s["speaker"] = "A"
    engine = f"qwen3_{a.clone}" if a.tts == "qwen3" else "xtts"
    tts_dir = w / f"tts_{engine}"  # one cache per engine and mode, so they can be compared
    step(f"Generating German voices ({f'Qwen3-TTS, {a.clone}' if a.tts == 'qwen3' else 'XTTS v2'})")
    synthesize(segs, samples, tts_dir, a.tts, a.batch)
    if a.stop_after == "tts":
        return

    step("Assembling, mixing, muxing")
    total = sf.info(str(audio)).duration
    late = assemble(segs, tts_dir, w / "voice_de.wav", total, a.max_speedup)
    for i, over in late:
        print(f'  ! line {i} runs {over:.1f}s late, consider shortening: {segs[i]["de"]}')
    mix(bg, w / "voice_de.wav", w / "mix.wav", a.bg_volume)
    mux(a.video, w / "mix.wav", out)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
