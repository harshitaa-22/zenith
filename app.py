from flask import Flask, request, jsonify, send_file
import base64
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import json
from datetime import datetime
import speech_recognition as sr
import pandas as pd
from datetime import datetime, timedelta
import requests


app = Flask(__name__)

# OpenAI API Key (Set as an Environment Variable for Security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINTEREST_TOKEN = os.getenv("PINTEREST_TOKEN")

CSV_FILE = 'tasks.csv'

# Load tasks from CSV or create empty DataFrame
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=["date", "start_time", "task", "done_status"])

# Ensure correct data types
df["done_status"] = df["done_status"].fillna(False).astype(bool)


if not OPENAI_API_KEY:
    raise ValueError("Set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

cached_response_vision = None
cached_response_vision = {"goals":[{"goal_heading":"Self-Care","goal":"Achieve clear skin and shiny hair","period_in_days":"90"},{"goal_heading":"Fitness","goal":"Develop a fit body through exercise","period_in_days":"60"},{"goal_heading":"Mental Wellness","goal":"Cultivate a healthy mind through mindfulness","period_in_days":"30"},{"goal_heading":"Nutrition","goal":"Incorporate healthy foods into diet","period_in_days":"30"},{"goal_heading":"Hydration","goal":"Drink more healthy smoothies","period_in_days":"30"}]}
cached_response_vision = {"goals": [{"goal": "Achieve clear skin, shiny hair, fit body, and a healthy mind","goal_heading": "Health and Fitness","period_in_days": "90"},{"goal": "Focus on self-love practices","goal_heading": "Self-Care","period_in_days": "30"},{"goal": "Incorporate more fruits and healthy meals","goal_heading": "Nutrition","period_in_days": "60"},{"goal": "Establish a regular workout routine","goal_heading": "Exercise","period_in_days": "30"}]}

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
                        {
                            "role": "system",
                            "content": (
                                "You are a strict JSON generator. NEVER include any explanation, markdown, or text outside the JSON object. "
                                "Return ONLY a raw JSON object, without triple backticks or surrounding text."
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Analyze this vision board and return goals in this format:\n"
                                        '{"goals":[{"goal_heading":"","goal":"","period_in_days":""}]}\n'
                                        "Return ONLY the raw JSON object. No text or formatting."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                                }
                            ]
                        }
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


@app.route("/tasks", methods=["GET"])
def get_tasks():
    year = request.args.get("year")
    month = request.args.get("month")

    if not (year and month):
        return jsonify({"error": "Please provide both 'year' and 'month' parameters"}), 400

    filtered = df[df["date"].str.contains(f"/{str(month).zfill(2)}/{year}")]
    return jsonify(filtered.to_dict(orient="records"))

@app.route("/task", methods=["POST"])
def add_task():
    global df
    data = request.get_json()

    if not all(k in data for k in ("date", "start_time", "task")):
        return jsonify({"error": "Missing fields in task data"}), 400

    task_entry = {
        "date": data["date"],
        "start_time": float(data["start_time"]),
        "task": data["task"],
        "done_status": False
    }

    df = pd.concat([df, pd.DataFrame([task_entry])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return jsonify({"message": "Task added"}), 201


@app.route("/task", methods=["PUT"])
def update_task():
    global df
    data = request.get_json()

    if not all(k in data for k in ("date", "start_time", "task")):
        return jsonify({"error": "Missing fields for identifying task"}), 400

    mask = (
        (df["date"] == data["date"]) &
        (df["start_time"] == float(data["start_time"])) &
        (df["task"] == data["task"])
    )

    if not mask.any():
        return jsonify({"error": "Task not found"}), 404

    df.loc[mask, "done_status"] = True
    df.to_csv(CSV_FILE, index=False)
    return jsonify({"message": "Task marked as done"})

@app.route("/enroll-challenge", methods=["POST"])
def enroll_to_goal():
    global df
    data = request.get_json()

    if not all(k in data for k in ("goal", "days")):
        return jsonify({"error": "Missing 'goal' or 'days' field"}), 400

    try:
        days = int(data["days"])
    except ValueError:
        return jsonify({"error": "'days' must be an integer"}), 400

    today = datetime.today()
    new_tasks = []

    for i in range(days):
        task_date = (today + timedelta(days=i)).strftime("%d/%m/%Y")
        new_tasks.append({
            "date": task_date,
            "start_time": 10.0,
            "task": data["goal"],
            "done_status": False
        })

    df = pd.concat([df, pd.DataFrame(new_tasks)], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return jsonify({"message": f"{days} tasks added for goal '{data['goal']}'"}), 201

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])

    # build messages list for chat
    messages = [{"role": "system", "content": "You are a helpful assistant. Respond in upto 100 words."}]
    for turn in history:
        messages.append({"role": "user", "content": turn["query"]})
        messages.append({"role": "assistant", "content": turn["response"]})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        ai_reply = response.choices[0].message.content.strip()
        return jsonify({"response": ai_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_vision_board", methods=["GET"])
def get_vision_board():
    board_id = request.args.get("board_id")
    if not board_id:
        return jsonify({"error": "Missing board_id"}), 400

    headers = {
        "Authorization": f"Bearer {PINTEREST_TOKEN}"
    }

    response = requests.get(
        f"https://api-sandbox.pinterest.com/v5/boards/{board_id}/pins",
        headers=headers
    )

    if response.status_code == 200:
        pins = response.json().get("items", [])
        simplified = [
            {
                "title": pin.get("title"),
                "description": pin.get("description"),
                "image_url": pin.get("media", {}).get("images", {}).get("original", {}).get("url")
            }
            for pin in pins
        ]
        return jsonify({"pins": simplified})
    else:
        return jsonify({"error": "Failed to fetch pins", "details": response.text}), 500

if __name__ == '__main__':
    app.run(debug=True)
