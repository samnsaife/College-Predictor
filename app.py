"""
app.py
------
Flask web application for the College Predictor ML system.
Run:  python app.py
      (make sure model.pkl exists — run model/train_model.py first)

Author: Sami Noor Saifi
"""

import os
import sys

from flask import Flask, jsonify, render_template, request

# ── Ensure model/ is importable ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from predictor import get_all_facilities, get_all_states, get_stats, predict_colleges

app = Flask(__name__)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    states     = get_all_states()
    facilities = get_all_facilities()
    stats      = get_stats()
    return render_template(
        "index.html",
        states=states,
        facilities=facilities,
        stats=stats,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        budget      = float(data.get("budget", 500000))
        state       = data.get("state", "Any")
        ctype       = data.get("college_type", "Any")
        gender      = data.get("gender", "Any")
        min_rating  = float(data.get("min_rating", 0.0))
        facilities  = data.get("facilities", [])
        top_n       = int(data.get("top_n", 5))

        results = predict_colleges(
            budget=budget,
            preferred_state=state,
            college_type=ctype,
            gender=gender,
            min_rating=min_rating,
            required_facilities=facilities,
            top_n=top_n,
        )

        return jsonify({"success": True, "colleges": results})

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
