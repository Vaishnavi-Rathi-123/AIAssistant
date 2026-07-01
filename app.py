from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import time

app = Flask(__name__)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

def call_hf_api(payload):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=10  # IMPORTANT
        )
        return response.json()

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Try again."}

def safe_api_call(payload):
    for attempt in range(3):
        try:
            return call_hf_api(payload)
        except:
            time.sleep(1)

    return {"error": "AI service unavailable right now"}

cache = {}

def get_response(prompt):
    if prompt in cache:
        return cache[prompt]

    response = safe_api_call(prompt)
    cache[prompt] = response

    return response

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.form["question"]

    response = client.chat.completions.create(
        model="google/gemma-3-1b-it",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI Personal Assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=150
    )

    answer = response.choices[0].message.content

    return jsonify({"response": answer})

@app.route("/summarize", methods=["POST"])
def summarize_email():

    email_text = request.form["email"]

    response = client.chat.completions.create(
        model="google/gemma-3-1b-it",
        messages=[
            {
                "role": "system",
                "content": "Summarize the email in 2-3 sentences."
            },
            {
                "role": "user",
                "content": email_text
            }
        ],
        max_tokens=150
    )

    answer = response.choices[0].message.content

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
