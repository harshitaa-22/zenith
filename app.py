from flask import Flask, request, jsonify, send_file
import base64
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import json
from datetime import datetime
import speech_recognition as sr


app = Flask(__name__)

# OpenAI API Key (Set as an Environment Variable for Security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

cached_response_vision = None
cached_response_vision = {"goals":[{"goal_heading":"Self-Care","goal":"Achieve clear skin and shiny hair","period_in_days":"90"},{"goal_heading":"Fitness","goal":"Develop a fit body through exercise","period_in_days":"60"},{"goal_heading":"Mental Wellness","goal":"Cultivate a healthy mind through mindfulness","period_in_days":"30"},{"goal_heading":"Nutrition","goal":"Incorporate healthy foods into diet","period_in_days":"30"},{"goal_heading":"Hydration","goal":"Drink more healthy smoothies","period_in_days":"30"}]}

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze-vision', methods=['POST'])
def analyze_image():
    """
    Accepts an image file, sends it to OpenAI Vision API, and returns a structured JSON response.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']

    # Convert image to base64
    image = Image.open(image_file)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if cached_response_vision:
        return cached_response_vision
    
    # OpenAI API Request
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that strictly outputs valid JSON without any extra text."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this vision board and generate goals in valid JSON format:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]}
            ],
            max_tokens=500
        )

        response_text = completion.choices[0].message.content.strip()

        # Extract valid JSON
        try:
            parsed_json = json.loads(response_text)
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse JSON from OpenAI response.", "raw_response": response_text}), 500

        return jsonify(parsed_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Dictionary mapping playlist names to file paths
audio_files = {
    "Peaceful Piano": "audio/peaceful_piano.mp3",
    "Soft Ambient": "audio/soft_ambient.mp3",
    "Nature Sounds": "audio/nature_sounds.mp3",
    "Upbeat Pop": "audio/upbeat_pop.mp3",
    "Feel Good Hits": "audio/feel_good_hits.mp3",
    "Summer Breeze": "audio/summer_breeze.mp3",
    "Deep Focus": "audio/deep_focus.mp3",
    "Study Beats": "audio/study_beats.mp3",
    "Concentration": "audio/concentration.mp3"
}

@app.route("/audio/<name>", methods=["GET"])
def get_audio(name):
    if name not in audio_files:
        return jsonify({"error": "Audio file not found"}), 404
    file_path = audio_files[name]
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist on server"}), 404
    return send_file(file_path, mimetype="audio/mpeg", as_attachment=True, download_name=f"{name}.mp3")

def transcribe_audio_file(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    print(f"Transcribed Text: {text}")
    return text

@app.route("/upload-journal", methods=["POST"])
def upload_voice_journal():
    text = request.form.get("text")
    audio = request.files.get("audio")

    if not audio:
        return jsonify({"error": "Missing text or audio"}), 400

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    text_filename = f"journal_{timestamp}.txt"
    audio_filename = f"journal_{timestamp}.wav"

    text_file_path = os.path.join(UPLOAD_FOLDER, text_filename)
    audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

    audio.save(audio_path)

    if not text:
        transcripted_text = transcribe_audio_file(audio_path)
        print(transcripted_text)
        text = transcripted_text
    
    with open(text_file_path, "w") as f:
        f.write(text)

    return jsonify({"message": "Journal saved successfully", "text_file": text_filename, "audio_file": audio_filename}), 200

@app.route("/journals", methods=["GET"])
def get_journals():
    journals = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith(".txt"):
            audio_filename = filename.replace(".txt", ".wav")
            journals.append({"text_file": filename, "audio_file": audio_filename})
    return jsonify(journals), 200

@app.route("/journal/audio/<filename>", methods=["GET"])
def get_journal_audio(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Audio file not found"}), 404
    return send_file(file_path, mimetype="audio/wav", as_attachment=True)

@app.route("/journal/text/<filename>", methods=["GET"])
def get_text(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Text file not found"}), 404
    with open(file_path, "r") as f:
        content = f.read()
    return jsonify({"text": content}), 200


if __name__ == '__main__':
    app.run(debug=True)
