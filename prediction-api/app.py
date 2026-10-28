
import os

from flask import Flask, request, jsonify

from winrate_predictor import WinratePredictor

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/winrate_predictor', methods=['POST'])  # path of the endpoint. Accept only HTTP POST requests
def predict_str():
    # the prediction input data in the message body as a JSON payload
    prediction_inout = request.get_json(silent=True)

    if not isinstance(prediction_inout, list) or len(prediction_inout) != 1 or not isinstance(prediction_inout[0], dict):
        return jsonify({'error': 'Invalid JSON input. Expected a list containing one match record.'}), 400

    try:
        status = dp.predict_single_record(prediction_inout)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400

    # return the prediction outcome as a json message. 200 is HTTP status code 200, indicating successful completion
    return jsonify(status), 200


dp = WinratePredictor()
# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.getenv("PORT", 5000)), host='0.0.0.0', debug=True)
