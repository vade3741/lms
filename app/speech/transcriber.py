from pathlib import Path
from faster_whisper import WhisperModel


AUDIO_DIR = Path("app/speech/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE = "base"


def save_audio_file(audio_bytes, file_name="candidate_answer.wav"):
    file_path = AUDIO_DIR / file_name

    with open(file_path, "wb") as file:
        file.write(audio_bytes)

    return str(file_path)


def transcribe_audio(audio_path):
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    transcript_parts = []

    for segment in segments:
        transcript_parts.append(segment.text)

    transcript = " ".join(transcript_parts).strip()

    return transcript