from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def get_explanation(prediction, confidence):
    if prediction == "FAKE":
        return "This news may contain misleading or suspicious language patterns commonly found in fake news."
    else:
        return "This news appears to follow more reliable and natural language patterns."

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    explanation = None

    if request.method == "POST":
        news = request.form["news"]

        vector = vectorizer.transform([news])
        prediction = model.predict(vector)[0]

        try:
            scores = model.decision_function(vector)
            confidence = round(abs(scores[0]) / (abs(scores[0]) + 1) * 100, 2)
        except:
            confidence = 85.0

        result = prediction
        explanation = get_explanation(result, confidence)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)
