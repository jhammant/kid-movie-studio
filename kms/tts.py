"""Text to speech, locally: macOS `say`, else espeak-ng (Linux, Windows), else a clear error."""
import shutil
import subprocess
from pathlib import Path

from kms import media

# macOS voice -> espeak-ng voice with a similar feel
ESPEAK_VOICES = {"Daniel": "en-gb+m3", "Fred": "en-us+m1", "Ralph": "en-us+m7", "Alex": "en-us+m2"}


def engine():
    if shutil.which("say"):
        return "say"
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            return name
    return None


def _say_has_voice(voice):
    r = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    return any(line.split()[0] == voice for line in r.stdout.splitlines() if line.strip())


def speak(text, out_wav, voice="Daniel", rate=110):
    """Speak text into a 48 kHz mono wav. `rate` is words per minute."""
    out_wav = Path(out_wav)
    eng = engine()
    if eng is None:
        raise RuntimeError("No text-to-speech found. Macs have `say` built in; on Linux install "
                           "espeak-ng (`sudo apt install espeak-ng`).")
    raw = out_wav.with_suffix(".raw.aiff" if eng == "say" else ".raw.wav")
    if eng == "say":
        cmd = ["say", "-r", str(rate), "-o", str(raw)]
        if _say_has_voice(voice):
            cmd += ["-v", voice]
        subprocess.run(cmd + [text.replace("[[", "[[").strip()], check=True)
    else:
        text = text.replace("[[slnc 900]]", ", , ,")  # espeak has no say-style silence command
        subprocess.run([eng, "-v", ESPEAK_VOICES.get(voice, "en"), "-s", str(rate), "-w", str(raw), text],
                       check=True)
    media.run([media.ffmpeg(), "-v", "error", "-y", "-i", raw, "-ac", "1", "-ar", str(media.SR), out_wav])
    raw.unlink(missing_ok=True)
    return out_wav
