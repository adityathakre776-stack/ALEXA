from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import time

app = Flask(__name__)

# =====================================================
# WHISPER
# =====================================================

print("=" * 50)
print("        VOICE RECOGNITION SERVER")
print("=" * 50)

print("Loading Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded!")
print("Server ready!")
print()

# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():
    return "ESP32 Voice Recognition Server Running"


# =====================================================
# TRANSCRIBE
# =====================================================

@app.route(
    "/transcribe",
    methods=["POST"]
)
def transcribe():

    print()
    print("=" * 50)
    print("🎤 AUDIO RECEIVED")
    print("=" * 50)

    audio = request.get_data()

    print(
        "Received bytes:",
        len(audio)
    )

    if len(audio) < 1000:

        return jsonify({
            "success": False,
            "text": "",
            "error": "Audio too small"
        }), 400

    filename = (
        "voice_"
        + str(int(time.time()))
        + ".wav"
    )

    try:

        with open(
            filename,
            "wb"
        ) as f:

            f.write(audio)

        print(
            "Saved:",
            filename
        )

        print()
        print(
            "🧠 Whisper processing..."
        )

        segments, info = model.transcribe(

            filename,

            language="en",

            beam_size=5,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 500
            },

            condition_on_previous_text=False
        )

        text_parts = []

        for segment in segments:

            text_parts.append(
                segment.text.strip()
            )

        text = " ".join(
            text_parts
        ).strip()

        print()
        print("=" * 50)
        print("📝 RECOGNIZED:")
        print("=" * 50)
        print(text)
        print("=" * 50)

        # Delete file

        try:
            os.remove(filename)
        except:
            pass

        return jsonify({
            "success": True,
            "text": text
        })

    except Exception as e:

        print(
            "❌ Whisper error:",
            str(e)
        )

        return jsonify({
            "success": False,
            "text": "",
            "error": str(e)
        }), 500


# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":

    print(
        "Starting Flask server..."
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
