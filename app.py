from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

# pipe = pipeline(
#     "text-generation",
#     model="google/gemma-3-1b-it",
#     device_map="auto"
# )

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
