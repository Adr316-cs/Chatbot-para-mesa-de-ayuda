# app.py
from flask import Flask, render_template, request, jsonify
from chatbot import predict_class, get_response, intents

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    intents_list = predict_class(user_message)
    response = get_response(intents_list, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)