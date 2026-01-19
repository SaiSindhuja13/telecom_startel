from flask import Flask, request, jsonify
from flask_cors import CORS
from hybrid_assistant import hybrid_answer
import analytics

app = Flask(__name__)
CORS(app)

@app.route("/api/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    response = hybrid_answer(question)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
