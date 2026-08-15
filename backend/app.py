import os
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# ============================================================
# LOAD V0.3.1 MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "scam_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "vectorizer.pkl"
)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        f"Vectorizer not found: {VECTORIZER_PATH}"
    )


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("✅ SafeLink v0.3.1 AI model loaded successfully!")


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(category, score):

    if category == "Safe":
        return "LOW"

    if score >= 80:
        return "HIGH"

    if score >= 50:
        return "MEDIUM"

    return "LOW"


# ============================================================
# SCAM DETECTION API
# ============================================================

@app.route("/api/detect", methods=["POST"])
def detect():

    try:

        # ----------------------------------------------------
        # GET REQUEST DATA
        # ----------------------------------------------------

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        text = str(
            data.get("text", "")
        ).strip()

        if not text:
            return jsonify({
                "error": "No text provided"
            }), 400

        # ----------------------------------------------------
        # TRANSFORM TEXT
        # ----------------------------------------------------

        x = vectorizer.transform([text])

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(x)[0]

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        probabilities = model.predict_proba(x)[0]

        max_probability = probabilities.max()

        score = round(
            max_probability * 100,
            2
        )

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        risk = get_risk_level(
            prediction,
            score
        )

        # ----------------------------------------------------
        # TOP 3 PREDICTIONS
        # ----------------------------------------------------

        classes = model.classes_

        ranked = sorted(
            zip(
                classes,
                probabilities
            ),
            key=lambda item: item[1],
            reverse=True
        )

        top_predictions = []

        for category_name, probability in ranked[:3]:

            top_predictions.append({
                "category": category_name,
                "confidence": round(
                    probability * 100,
                    2
                )
            })

        # ----------------------------------------------------
        # TERMINAL LOG
        # ----------------------------------------------------

        print("\n===================================")
        print("SafeLink v0.3.1 Detection")
        print("===================================")
        print("Input      :", text)
        print("Prediction :", prediction)
        print("Confidence :", score, "%")
        print("Risk       :", risk)
        print("Top 3      :", top_predictions)
        print("===================================\n")

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({
            "category": prediction,
            "score": score,
            "risk": risk,
            "top_predictions": top_predictions
        })

    except Exception as e:

        print(
            "❌ Detection error:",
            str(e)
        )

        return jsonify({
            "error": "Detection failed"
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "SafeLink v0.3.1"
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "🚀 SafeLink backend running on "
        "http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )