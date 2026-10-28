
# importing Flask and other modules
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        prediction_input = [
            {
                "league": request.form.get("league"),
                "elo_home": int(request.form.get("elo_home")),
                "elo_away": int(request.form.get("elo_away")),
                "crowd_present": int(request.form.get("crowd_present")),
                "stadium_capacity": int(request.form.get("stadium_capacity")),
                "attendance_pct": float(request.form.get("attendance_pct")),
                "travel_distance_km": float(request.form.get("travel_distance_km")),
                "timezones_crossed": int(request.form.get("timezones_crossed")),
                "is_derby": int(request.form.get("is_derby")),
                "venue_familiarity": int(request.form.get("venue_familiarity"))
            }
        ]

        app.logger.debug("Prediction input : %s", prediction_input)

        # Send an HTTP POST request to the prediction API
        predictor_api_url = os.environ['PREDICTOR_API']
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        prediction_output = res.json()
        prediction_value = prediction_output['prediction']
        probabilities = prediction_output['probabilities']

        app.logger.info("Prediction Output : %s", prediction_value)

        return render_template("response_page.html",
                               prediction_variable=prediction_value,
                               probabilities=probabilities)

    else:
        return jsonify(message="Method Not Allowed"), 405


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
