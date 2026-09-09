from flask import Flask, jsonify, request
from mental_wellbeing.predict import predict_student

app = Flask(__name__)

@app.get("/")
def index():
    return jsonify({
        "service": "Student Wellbeing Intelligence API",
        "status": "running",
        "routes": {
            "health": "GET /health",
            "predict": "POST /predict",
        },
        "dashboard": "Run `py run_dashboard.py` and open the Streamlit URL for the visual application.",
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/favicon.ico")
def favicon():
    return "", 204

@app.post("/predict")
def predict():
    try:
        return jsonify(predict_student(request.get_json(force=True), include_explanations=True))
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(debug=True)
