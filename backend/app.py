import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "scam_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("✅ SafeLink AI model loaded successfully!")


# ============================================================
# SCAM DETECTION API
# ============================================================

@app.route("/api/detect", methods=["POST"])
def detect():

    try:

        # Get JSON request
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        # Get text
        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "error": "No text provided"
            }), 400

        # ====================================================
        # TRANSFORM TEXT
        # ====================================================

        x = vectorizer.transform([text])

        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(x)[0]

        # ====================================================
        # CONFIDENCE
        # ====================================================

        probabilities = model.predict_proba(x)[0]

        max_probability = probabilities.max()

        score = round(max_probability * 100, 2)

        # ====================================================
        # CATEGORY
        # ====================================================

        # For now, DON'T use the uncertainty threshold.
        # We want to see what the model actually predicts.

        category = prediction

        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        classes = model.classes_

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        top_predictions = []

        for category_name, probability in ranked[:3]:

            top_predictions.append({
                "category": category_name,
                "confidence": round(probability * 100, 2)
            })

        # ====================================================
        # DEBUG TERMINAL OUTPUT
        # ====================================================

        print("\n===================================")
        print("SafeLink Detection")
        print("===================================")
        print("Input      :", text)
        print("Prediction :", category)
        print("Confidence :", score, "%")
        print("Top 3      :", top_predictions)
        print("===================================\n")

        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({
            "category": category,
            "score": score,
            "top_predictions": top_predictions
        })

    except Exception as e:

        print("❌ Detection error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("🚀 SafeLink backend running on http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )