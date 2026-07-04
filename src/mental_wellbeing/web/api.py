from __future__ import annotations

from flask import Flask, jsonify, request

from mental_wellbeing.predict import predict_depression_risk, predict_mental_status

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict/mental-status")
def mental_status():
    return jsonify(predict_mental_status(request.get_json(force=True)))


@app.post("/predict/depression-risk")
def depression_risk():
    return jsonify(predict_depression_risk(request.get_json(force=True)))


if __name__ == "__main__":
    app.run(debug=True)
