import io
import tempfile
from pathlib import Path
import streamlit as st

def text_to_speech_bytes(text):
    """
    Generates audio bytes for the given text using gTTS (with pyttsx3 fallback).
    Returns (bytes, mime_type) or (None, None).
    """
    if not text or not str(text).strip():
        return None, None

    clean_text = str(text).strip()

    # Try gTTS first (higher quality voice)
    try:
        from gtts import gTTS
        tts = gTTS(text=clean_text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read(), "audio/mp3"
    except Exception:
        pass

    # Try pyttsx3 offline fallback
    try:
        import pyttsx3
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
        engine.save_to_file(clean_text, temp_path)
        engine.runAndWait()
        with open(temp_path, "rb") as f:
            data = f.read()
        Path(temp_path).unlink(missing_ok=True)
        return data, "audio/wav"
    except Exception:
        pass

    return None, None


def render_tts_button(text, button_label="🔊 Read Question Aloud", key="tts_btn"):
    """
    Renders an audio player and a Web Speech API button for instant audio playback.
    """
    if not text or not str(text).strip():
        return

    # Escaping text for JS
    safe_text = str(text).replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", " ")

    html_code = f"""
    <div style="margin-top: 5px; margin-bottom: 10px;">
        <button id="speak_btn_{key}" onclick="speakText_{key}()" style="
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 8px 16px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            transition: 0.2s;
        ">
            {button_label}
        </button>
        <button id="stop_btn_{key}" onclick="stopSpeech_{key}()" style="
            background-color: #f44336;
            border: none;
            color: white;
            padding: 8px 12px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
            margin-left: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        ">
            ⏹️ Stop
        </button>
    </div>

    <script>
        function speakText_{key}() {{
            window.speechSynthesis.cancel(); // Stop any ongoing speech
            var msg = new SpeechSynthesisUtterance("{safe_text}");
            msg.rate = 1.0;
            msg.pitch = 1.0;
            msg.lang = 'en-US';
            window.speechSynthesis.speak(msg);
        }}

        function stopSpeech_{key}() {{
            window.speechSynthesis.cancel();
        }}
    </script>
    """
    st.components.v1.html(html_code, height=55)
