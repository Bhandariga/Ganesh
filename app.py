from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load HuggingFace model
chatbot = pipeline("text-generation", model="gpt2")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data["message"]
    response = chatbot(user_input, max_length=100, do_sample=True)[0]['generated_text']
    return jsonify({"reply": response.strip()})

if __name__ == "__main__":
    app.run(debug=True)
