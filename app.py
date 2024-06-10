from flask import Flask, render_template, request, jsonify, send_from_directory
from recommend import generate_recommendations
from questions import questions
import time

app = Flask(__name__)
current_question_index = 0
user_responses = {}
feedback_stage = False


@app.route("/")
def start():
    global current_question_index, user_responses, feedback_stage
    current_question_index = 0
    user_responses = {}
    feedback_stage = False
    return render_template("start.html")


@app.route("/chat")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chat():
    global current_question_index, user_responses, feedback_stage
    msg = request.form.get("msg")

    if feedback_stage:
        if msg.lower() == "yes":
            feedback_stage = False
            return jsonify(
                {"response": "I'm delighted to hear that I've been helpful. Have a great day!", "end_chat": True})
        elif msg.lower() == "no":
            current_question_index = 0
            user_responses = {}
            feedback_stage = False
            question = questions[current_question_index]
            response = {
                "question": question["question"],
                "type": question["type"],
                "choices": question.get("choices", [])
            }
            current_question_index += 1
            return jsonify(response)

    if msg and current_question_index > 0:
        question_key = questions[current_question_index - 1]["key"]
        user_responses[question_key] = msg

    if current_question_index < len(questions):
        question = questions[current_question_index]
        response = {
            "question": question["question"],
            "type": question["type"],
            "choices": question.get("choices", [])
        }
        current_question_index += 1
    else:
        try:
            user_city = user_responses.pop("city", "default_city")
            user_budget = float(user_responses.pop("budget", 0))
            response = {
                "response": "Looking for destinations..."
            }
            time.sleep(2)
            recommendations = generate_recommendations(user_responses, user_city, user_budget)
            if not recommendations:
                raise ValueError("Recommendations could not be found.")
            response = {
                "response": "Best recommendations based on your responses are: ",
                "recommendations": recommendations,
                "feedback": "Were these recommendations useful?"
            }
            feedback_stage = True
        except Exception as e:
            response = {
                "response": "Recommendations could not be found. Please try again.",
                "error": "Recommendations could not be found. Please try again."
            }
            current_question_index = 0  # reset the questions index
            user_responses = {}  # reintialize user responses
            feedback_stage = False  # reset feedback

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
