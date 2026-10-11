# dub: local AI video dubbing, English -> German

Turns an English video into a German dub. Every speaker gets their own cloned voice. Music and sound effects stay in the mix. The original audio stays as a second track. Everything runs on your own AMD Radeon GPU: nothing goes to a cloud service.

With an AMD Radeon RX 7800 XT (16 GB), a 22-minute video takes about 15 minutes once the models are downloaded.

I built it to dub science shows for my kids, it is not meant to take work from professional voice actors :)

## What it does

1. **Separates speech from music and effects** ([Demucs](https://github.com/adefossez/demucs)). The German voices are mixed over the original background.
2. **Transcribes** the English with the time of every word ([Whisper](https://github.com/openai/whisper)) and **finds who speaks when** ([pyannote](https://github.com/pyannote/pyannote-audio)).
3. **Cuts the speech into lines** of whole sentences, one speaker each. Each line's voice is compared with every speaker's voiceprint. This corrects lines that the speaker detection assigns to the wrong person in quick exchanges.
4. **Translates** with a local LLM ([Gemma 4](https://ollama.com/library/gemma4) via [Ollama](https://ollama.com)):
   - It first reads the whole transcript and writes a summary, a description of each speaker and a glossary of names and terms.
   - It then translates each line with that context, the previous lines' translations and the speaker's role and gender.
   - A second pass checks every line for changed numbers or names, broken words and English leftovers.
5. **Speaks German in each speaker's cloned voice** ([Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)). Only the timbre is cloned, so the German has native pronunciation instead of an English accent.
6. **Builds the output video.** Every line starts at its original time and is sped up slightly if the German runs longer. The voices are mixed over the background. The video gets the German track as default plus all original audio tracks.

Every step caches its result in `work/`. After you edit the transcript, speakers or translation, a new run redoes only what changed.

### Two versions

| | `dub_v2.py` | `dub_v1.py` |
|---|---|---|
| Voices | One per speaker | One for everyone, cloned from the longest lines |
| Lines | Whole sentences, cut at speaker changes | As Whisper splits them, sometimes mid-sentence |
| Input | Any video. With several audio tracks, it dubs the English one | Videos with one audio track |
| Extra setup | pyannote and a free Hugging Face account (step 5) | None |

## Requirements

- **Ubuntu 24.04** or a derivative such as Linux Mint 22. The stock `amdgpu` kernel driver is enough: PyTorch and Ollama bring their own ROCm runtime, so there is no separate ROCm install.
- **AMD Radeon RX 7000 series (RDNA 3) GPU with 16 GB VRAM.** `--batch 4` lowers the VRAM use of the voice step.
- **RAM:** at least 16 GB.
- **Disk:** about 30 GB (Python environment about 10 GB, models about 15 GB).
- **Internet** during setup and the first run, which download the models.

## Quick start

### 1. System packages and GPU access

```bash
sudo apt update
sudo apt install -y ffmpeg git curl build-essential python3-dev libsndfile1
sudo usermod -aG render,video $USER     # then log out and back in
```

### 2. uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # then open a new terminal
```

### 3. Get the code and install

```bash
git clone https://github.com/xtavras/dub.git
cd dub
uv sync --extra speakers     # dub_v1.py alone: plain uv sync
```

This downloads several GB (PyTorch for ROCm). Check that PyTorch sees the GPU. AMD GPUs show up as `cuda` in PyTorch:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.11.0+rocm7.2 True AMD Radeon RX 7...
```

### 4. Ollama and the translation model

```bash
curl -fsSL https://ollama.com/install.sh | sh     # detects AMD, installs the ROCm build
ollama pull gemma4:12b                            # about 8 GB
```

While `ollama run gemma4:12b --think=false "Hallo"` answers, `ollama ps` shows `100% GPU` if the model runs on the GPU.

### 5. Hugging Face token for speaker detection (`dub_v2.py` only)

pyannote's speaker-detection model is free but gated. You accept its terms once and log in with **your own** token:

1. Create a free account at <https://huggingface.co/join>.
2. Open <https://huggingface.co/pyannote/speaker-diarization-community-1> and accept the conditions.
3. Create a token at <https://huggingface.co/settings/tokens>: **Create new token** -> type **Read** -> copy it (it starts with `hf_`).
4. Log in and paste the token when asked:

   ```bash
   uv run hf auth login
   ```

   The token is stored in `~/.cache/huggingface/token`, outside this repository. A token must not go into any file in the repository. For a single shell session, `export HF_TOKEN=hf_...` works as well.
5. Check access:

   ```bash
   uv run python -c "from huggingface_hub import hf_hub_download as d; d('pyannote/speaker-diarization-community-1', 'config.yaml'); print('access OK')"
   ```

### 6. Dub a video

```bash
uv run dub_v2.py /path/to/video.mp4
# -> /path/to/video_de.mp4 (MKV in -> MKV out)
```

The first run also downloads the Whisper, Demucs, Qwen3-TTS and pyannote models (about 7 GB). The script prints each step, the speakers it finds and the lines that run late.

## Review before the voice step

`--stop-after translate` stops before any voice is generated. Edit the files below, then run the same command again:

```bash
uv run dub_v2.py video.mp4 --stop-after translate
# check and edit the files below
uv run dub_v2.py video.mp4      # continues; voices only changed lines
```

Files in `work/<video-name>/v2/` (`dub_v1.py`: `work/<video-name>/`):

| File | What you can fix |
|---|---|
| `context.json` | The description of each speaker and the glossary, e.g. names Whisper misheard. Changes apply to lines translated afterwards. To translate everything again, delete `segments_de.json` |
| `segments_de.json` | `"de"`: the German line (shorten long ones). `"speaker"`: the voice that speaks the line |
| `voices/voices.json` | `"gender"` per speaker, if the guess from the voice pitch is wrong |
| `voices/<letter>.wav` | Each speaker's voice sample. A sample with music or a second voice gives a worse clone |

## Options

| Option | Purpose |
|---|---|
| `--speakers 3` | Number of speakers, if you know it (default: detected) |
| `--speakers 1` | One voice for everyone: no speaker detection and no Hugging Face account |
| `--audio-track 1` | Audio track to dub, counted from 0 (default: the one tagged English) |
| `--address Sie` | Formal "Sie" (documentaries, business). Default: informal "du" |
| `--llm qwen3:14b` | A different Ollama model for the translation |
| `--whisper large-v3` | More accurate transcription (slower, more VRAM) |
| `--batch 4` | Smaller batches in the voice step, for less VRAM |
| `--ref voice.wav` | One clean voice sample (10-30 s) for every speaker |
| `--clone full` | Voices closer to the originals, but with an English accent |
| `--tts xtts` | XTTS v2 instead of Qwen3-TTS (non-commercial license, see below) |
| `--max-speedup 1.4` | Maximum speed-up for German lines that run longer than the English (default 1.3) |
| `--bg-volume 0.7` | Quieter music and effects under the German voices |
| `--stop-after translate` | Stop after a step: `separate`, `transcribe`, `translate` or `tts` |
| `-o out.mkv` | Output file (default: `<video>_de.<same type>` next to the input) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | `groups` must list `render` and `video` (log out and back in after step 1). If it is still `False`, or you see `invalid device function`: `export HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| Out of GPU memory | `--batch 4`. Close browsers and games. Other Ollama models must not be loaded (`ollama ps`, `ollama stop <model>`) |
| The run stops with `HW Exception ... GPU Hang`, the desktop may freeze | A crash in Ollama's ROCm backend. `journalctl -k` shows `page fault ... for process llama-server`, then a GPU reset, which stops every program on the GPU. Reboot if the desktop stays frozen, then run the same command again: finished steps are cached |
| `Speaker detection needs pyannote` | `uv sync --extra speakers`, or use `--speakers 1` |
| `Can't load pyannote/speaker-diarization-community-1` | Log in and accept the model's terms (step 5) |
| A line has the wrong voice | Change its `"speaker"` letter in `segments_de.json` and run again |
| One person has two voices (e.g. narrator and on camera) | Change one letter to the other in `segments_de.json`. Or delete `v2/diarization.json` and every file created after it, then run with `--speakers N` |
| Lines run late | Shorten them in `segments_de.json`, or raise `--max-speedup` a little |
| `sox: not found`, `flash-attn is not installed` | Qwen3-TTS prints these, but they don't affect the result. `sudo apt install sox` removes the first |
| PyTorch is a CUDA build (`+cu...`), or `nvidia-*` packages are installed | A `pip install` replaced locked packages. `uv sync --extra speakers` restores the locked environment. Add packages with `uv add` |

## Notes for AMD / ROCm

`pyproject.toml` pins the environment to versions that work on RDNA 3:

- **PyTorch 2.11 from PyTorch's ROCm 7.2 index.** torchaudio's last release is 2.11. It needs the same torch version.
- **torchcodec from PyTorch's CPU index.** The PyPI build links against CUDA and fails to load on AMD.
- **No CUDA `triton` for Whisper.** The override removes it because it clashes with the ROCm `triton-rocm` package.
- **transformers 4.57.3.** qwen-tts pins it. coqui-tts (XTTS) fails to import with 5.1 and later.
- **`MIOPEN_FIND_MODE=FAST`,** set by the scripts. Otherwise AMD's MIOpen benchmarks kernels for every new input length, which makes XTTS about 5x slower.
- **pyannote's LSTM runs without MIOpen.** MIOpen's LSTM kernels fail on gfx1101 with `miopenStatusUnknownError`.

## Limitations

- **English -> German only.** The prompts and language settings are written for that pair.
- **Very short lines** ("Yes.", "Cool.") are too short to tell speakers apart by voice and can get the wrong one. Fix them with their speaker letter.
- **Emotion carries over only in part.** The timbre is cloned, but the intonation is more neutral than the original.
- **Names Whisper mishears** go into the translation. Fix them in `context.json`.
- **German is about 20-30% longer than English.** Lines are sped up at most 1.3x, so a few can still run slightly late.

## Licenses

The code in this repository is under the [MIT License](LICENSE). The models it downloads have their own licenses:

| Model | License |
|---|---|
| [Qwen3-TTS 1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Apache 2.0 |
| [Gemma 4](https://ollama.com/library/gemma4) | Apache 2.0 |
| [Whisper](https://github.com/openai/whisper) | MIT |
| [Demucs](https://github.com/adefossez/demucs) | MIT |
| [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | CC-BY-4.0 (free, gated) |
| [XTTS v2](https://huggingface.co/coqui/XTTS-v2), only with `--tts xtts` | Coqui Public Model License: **non-commercial** |

Voice cloning copies real people's voices. Use it only on videos you have the rights to.
