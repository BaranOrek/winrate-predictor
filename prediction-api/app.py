import os

from flask import Flask, request, jsonify

from diabetes_predictor import DiabetesPredictor

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/diabetes_predictor', methods=['POST'])  # path of the endpoint. Except only HTTP POST request
def predict_str():
    # the prediction input data in the message body as a JSON payload
    prediction_inout = request.get_json()
    status = dp.predict_single_record(prediction_inout)
    # return the prediction outcome as a json message. 200 is HTTP status code 200, indicating successful completion
    return jsonify({'result': str(status)}), 200


dp = DiabetesPredictor()
# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.getenv("PORT", 5000)), host='0.0.0.0', debug=True)
