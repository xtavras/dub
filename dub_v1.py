#!/usr/bin/env python3
"""Dub an English video into German, fully local. Version 1: one voice for everyone.

Pipeline: ffmpeg -> Demucs (split voice/background) -> Whisper (transcribe)
-> Ollama LLM (translate) -> Qwen3-TTS or XTTS v2 (cloned German voice) -> ffmpeg (mix + mux).
Every step caches its result in the work dir, so you can edit and re-run.

Translation with whole-video context (context.json: summary + glossary) and a check pass for every
line; Qwen3-TTS in batches with voice-only cloning (--clone timbre), so the German has no English
accent. Needs no speaker detection and no Hugging Face account; dub_v2.py gives every speaker
their own voice.
"""
import argparse
import gc
import hashlib
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

CONTEXT_PROMPT = """Below is the English transcript of a video that will be dubbed into German.
Answer with a JSON object:
- "summary": 2-3 sentences: what the video is about, who it is for, its tone.
- "glossary": an object mapping every name (people, shows, places, organisations, missions,
  spacecraft, products) and technical term to how it should appear in German.
  Keep names unchanged; use the usual German term for everything else.

TRANSCRIPT:
{transcript}"""

PROMPT = """You translate English video dialogue into German for a voice-over dub.
About the video: {summary}
Glossary, use exactly these: {glossary}

Translate only the LINE. Rules:
- Natural spoken German, same meaning and tone. Address the audience with "{address}".
- It must be spoken in about {seconds:.1f} seconds, so keep it as short as the English; rephrase rather than add words.
- Keep every fact, name and number exactly. Write numbers as digits, like the English.
- Answer with the German line only: no quotes, no notes, no English.

Previous lines (English -> German):
{before}
Following lines (English, only for context): {after}

LINE: {line}"""

CHECK_PROMPT = """You review one line of a German voice-over dub against its English original.
About the video: {summary}
Glossary: {glossary}

English: {en}
German: {de}
{hint}
Check the German for: changed or missing facts, wrong numbers or names, misspelled or broken words,
leftover English (names and glossary terms are fine), wrong form of address (must be "{address}").
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


def extract_audio(video, out):
    run("ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "2", "-ar", "44100", out)


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
    import whisper

    model = whisper.load_model(model_name, device=DEV)
    res = model.transcribe(whisper.load_audio(str(vocals)), language="en",
                           fp16=DEV == "cuda", condition_on_previous_text=False)
    del model
    free_gpu()
    return [{"start": s["start"], "end": s["end"], "en": s["text"].strip()}
            for s in res["segments"] if s["text"].strip()]


def merge_segments(segs, max_len=12.0, max_gap=0.6):
    """Join Whisper fragments into full sentences: better translations, fewer cuts."""
    out = []
    for s in segs:
        p = out[-1] if out else None
        if (p and not p["en"].endswith((".", "?", "!"))
                and s["start"] - p["end"] <= max_gap and s["end"] - p["start"] <= max_len):
            p["end"], p["en"] = s["end"], f'{p["en"]} {s["en"]}'
        else:
            out.append(dict(s))
    return out


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
    """Read the whole transcript once: a summary and a glossary of names and terms for every line."""
    transcript = "\n".join(s["en"] for s in segs)
    # the default context window (4096 tokens) is too small for a whole episode; ~3 chars per token
    num_ctx = (len(CONTEXT_PROMPT + transcript) // 3 // 4096 + 2) * 4096
    try:
        ctx = json.loads(ollama(a.ollama_url, a.llm, CONTEXT_PROMPT.format(transcript=transcript),
                                fmt="json", num_ctx=num_ctx))
    except json.JSONDecodeError:
        ctx = {}
    return {"summary": str(ctx.get("summary", "")), "glossary": dict(ctx.get("glossary") or {})}


def glossary_for(ctx, text):
    """Only the glossary entries that occur in this text: keeps the prompts short."""
    hits = [f"{en} = {de}" for en, de in ctx["glossary"].items() if en.lower() in text.lower()]
    return "; ".join(hits) or "-"


def numbers(text):
    return sorted(re.sub(r"[.,]", "", n) for n in re.findall(r"\d[\d.,]*\d|\d", text))


def translate_line(segs, i, ctx, a):
    s = segs[i]
    before = "\n".join(f'{x["en"]} -> {x.get("de", "?")}' for x in segs[max(0, i - 3):i]) or "-"
    after = " ".join(x["en"] for x in segs[i + 1:i + 3]) or "-"
    return clean(ollama(a.ollama_url, a.llm, PROMPT.format(
        summary=ctx["summary"] or "-", glossary=glossary_for(ctx, f"{before} {s['en']} {after}"),
        address=a.address, seconds=s["end"] - s["start"], line=s["en"], before=before, after=after)))


def check_line(s, ctx, a):
    """Second opinion on one translated line. Returns the corrected line, or None if it's fine."""
    # only numbers the German lost: a German "4" for an English "four" is fine
    missing = [n for n in numbers(s["en"]) if n not in numbers(s["de"])]
    hint = f"Note: the German is missing the number(s) {', '.join(missing)} from the English.\n" if missing else ""
    answer = clean(ollama(a.ollama_url, a.llm, CHECK_PROMPT.format(
        summary=ctx["summary"] or "-", glossary=glossary_for(ctx, s["en"]), en=s["en"], de=s["de"],
        hint=hint, address=a.address)))
    if re.sub(r"\W", "", answer).upper() == "OK" or answer == s["de"]:
        return None
    # A real fix never drops a third of the line; that's the reviewer cutting content (it does so
    # on lines that start mid-sentence)
    return answer if len(answer) >= 0.7 * len(s["de"]) else None


def translate(segs, path, ctx, a):
    for i, s in enumerate(segs):
        if "de" not in s:
            s["de"] = translate_line(segs, i, ctx, a)
            print(f'[{i + 1}/{len(segs)}] {s["en"]}\n          -> {s["de"]}')
        if s["de"] and not s.get("checked"):  # once per line, so your own edits stay
            fixed = check_line(s, ctx, a)
            if fixed:
                print(f'  check fixed line {i}: {s["de"]}\n                   -> {fixed}')
                s["de_unchecked"], s["de"] = s["de"], fixed
            s["checked"] = True
        path.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
    ollama(a.ollama_url, a.llm, None, keep_alive=0)  # unload the LLM to free VRAM for the TTS model


def make_reference(vocals, segs, wav_out, txt_out, seconds=15):
    """Build a clean voice sample for cloning from the longest spoken segments.
    Whole segments only, so its transcript (Qwen3-TTS needs one) matches the audio exactly."""
    wav, sr = sf.read(vocals, dtype="float32", always_2d=True)
    mono, picked, total = wav.mean(1), [], 0.0
    for s in sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True):
        picked.append(s)
        total += s["end"] - s["start"]
        if total >= seconds:
            break
    picked.sort(key=lambda s: s["start"])
    sf.write(wav_out, np.concatenate([mono[int(s["start"] * sr):int(s["end"] * sr)] for s in picked]), sr)
    txt_out.write_text(" ".join(s["en"] for s in picked))


def clip_path(tts_dir, i, text):
    # the text hash in the name means edited lines are re-synthesized automatically
    return tts_dir / f"{i:04d}_{hashlib.md5(text.encode()).hexdigest()[:8]}.wav"


def synthesize(segs, ref, ref_text, tts_dir, engine, batch):
    tts_dir.mkdir(exist_ok=True)
    todo = [(i, s["de"]) for i, s in enumerate(segs)
            if s["de"] and not clip_path(tts_dir, i, s["de"]).exists()]
    if not todo:
        return
    if engine == "qwen3":
        synth_qwen3(todo, ref, ref_text, tts_dir, batch)
    else:
        synth_xtts(todo, ref, tts_dir)
    free_gpu()


def synth_xtts(todo, ref, tts_dir):
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEV)
    for n, (i, text) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {text}")
        wav = tts.tts(text=text, speaker_wav=str(ref), language="de")
        sf.write(clip_path(tts_dir, i, text), np.asarray(wav, dtype=np.float32), SR)


def synth_qwen3(todo, ref, ref_text, tts_dir, batch):
    import librosa
    from qwen_tts import Qwen3TTSModel
    from transformers.utils import logging

    logging.set_verbosity_error()  # hides "Setting `pad_token_id` to `eos_token_id`" on every batch
    tts = Qwen3TTSModel.from_pretrained(QWEN3_MODEL, device_map=f"{DEV}:0" if DEV == "cuda" else DEV,
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    # Built once, reused for every line. Without ref_text only the voice's timbre is cloned, so the
    # German comes out native; with it the model continues the English sample and keeps its accent.
    prompt = tts.create_voice_clone_prompt(ref_audio=str(ref), ref_text=ref_text,
                                           x_vector_only_mode=ref_text is None)
    todo = sorted(todo, key=lambda t: len(t[1]))  # similar lengths in a batch: less padding
    n = 0
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        texts = [text for _, text in chunk]
        # The batch runs until its longest line ends, so cap the length (~12.5 codec frames per
        # second, generously ~6 characters per second): one runaway line can't stall the batch.
        wavs, sr = tts.generate_voice_clone(text=texts, language=["german"] * len(chunk),
                                            voice_clone_prompt=prompt * len(chunk),
                                            max_new_tokens=int(12.5 * (max(map(len, texts)) / 6 + 5)))
        for (i, text), wav in zip(chunk, wavs):
            n += 1
            print(f"[{n}/{len(todo)}] {text}")
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            sf.write(clip_path(tts_dir, i, text), np.asarray(wav, dtype=np.float32), SR)


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
        clip, _ = sf.read(clip_path(tts_dir, i, s["de"]), dtype="float32")
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
    """German track as default, original English kept as a second audio track."""
    run("ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-metadata:s:a:0", "language=deu", "-metadata:s:a:1", "language=eng",
        "-disposition:a:0", "default", "-disposition:a:1", "0", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="default: <video>_de.mp4")
    ap.add_argument("--work", type=Path, default=Path("work"), help="cache dir (default: ./work)")
    ap.add_argument("--whisper", default="turbo", help="turbo (fast, default) or large-v3 (most accurate)")
    ap.add_argument("--llm", default="gemma4:12b", help="Ollama model for translation")
    ap.add_argument("--address", choices=["du", "Sie"], default="du", help="how to address the audience")
    ap.add_argument("--tts", choices=["qwen3", "xtts"], default="qwen3", help="TTS engine (default: qwen3)")
    ap.add_argument("--batch", type=int, default=8, help="lines per Qwen3-TTS batch (more = faster, more VRAM)")
    ap.add_argument("--ref", type=Path, help="your own clean voice sample (10-30 s WAV); default: auto")
    ap.add_argument("--clone", choices=["timbre", "full"], default="timbre",
                    help="Qwen3-TTS: timbre = voice only, native German (default); "
                         "full = also imitates the English sample: closer voice, but English accent")
    ap.add_argument("--ref-text", help="what is said in --ref (only for --clone full); default: transcribed")
    ap.add_argument("--max-speedup", type=float, default=1.3, help="max speed-up for long German lines")
    ap.add_argument("--bg-volume", type=float, default=1.0, help="music/SFX volume in the mix")
    ap.add_argument("--stop-after", choices=["separate", "transcribe", "translate", "tts"])
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    a = ap.parse_args()

    work = a.work / a.video.stem
    work.mkdir(parents=True, exist_ok=True)
    out = a.out or a.video.with_name(f"{a.video.stem}_de.mp4")
    print(f"Device: {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'CPU (no GPU found!)'}")

    audio, vocals, bg = work / "audio.wav", work / "vocals.wav", work / "background.wav"
    if not audio.exists():
        step("Extracting audio")
        extract_audio(a.video, audio)
    if not vocals.exists():
        step("Separating voice from music/SFX (Demucs)")
        separate(audio, vocals, bg)
    if a.stop_after == "separate":
        return

    seg_en, seg_de = work / "segments_en.json", work / "segments_de.json"
    if not seg_en.exists():
        step(f"Transcribing (Whisper {a.whisper})")
        seg_en.write_text(json.dumps(merge_segments(transcribe(vocals, a.whisper)), indent=1))
    if a.stop_after == "transcribe":
        return

    # segments_de.json wins once it exists, so your manual edits are kept
    segs = json.loads((seg_de if seg_de.exists() else seg_en).read_text())
    if any("de" not in s or (s["de"] and not s.get("checked")) for s in segs):
        ctx_path = work / "context.json"
        if not ctx_path.exists():  # like segments_de.json: edit it, and your version wins
            step(f"Reading the whole transcript ({a.llm})")
            ctx_path.write_text(json.dumps(analyze(segs, a), ensure_ascii=False, indent=1))
        step(f"Translating and checking ({a.llm})")
        translate(segs, seg_de, json.loads(ctx_path.read_text()), a)
    if a.stop_after == "translate":
        print(f"\nReview/edit {seg_de}, then run again without --stop-after.")
        return

    if a.ref:
        ref, ref_text = a.ref, a.ref_text
    else:
        ref, ref_txt = work / "reference.wav", work / "reference.txt"
        if not ref_txt.exists():
            make_reference(vocals, segs, ref, ref_txt)
        ref_text = ref_txt.read_text()
    full = a.tts == "qwen3" and a.clone == "full"
    if full and not ref_text:
        ref_text = " ".join(s["en"] for s in transcribe(ref, a.whisper))
    engine = f"qwen3_{a.clone}" if a.tts == "qwen3" else "xtts"
    tts_dir = work / f"tts_{engine}"  # one cache per engine and mode, so they can be compared
    step(f"Generating German voice ({f'Qwen3-TTS, {a.clone}' if a.tts == 'qwen3' else 'XTTS v2'})")
    synthesize(segs, ref, ref_text if full else None, tts_dir, a.tts, a.batch)
    if a.stop_after == "tts":
        return

    step("Assembling, mixing, muxing")
    total = sf.info(str(audio)).duration
    late = assemble(segs, tts_dir, work / "voice_de.wav", total, a.max_speedup)
    for i, over in late:
        print(f'  ! line {i} runs {over:.1f}s late, consider shortening: {segs[i]["de"]}')
    mix(bg, work / "voice_de.wav", work / "mix.wav", a.bg_volume)
    mux(a.video, work / "mix.wav", out)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
