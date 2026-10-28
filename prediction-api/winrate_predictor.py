import json
import os

import pandas as pd
from flask import jsonify
import logging
from io import StringIO
import pickle


class DiabetesPredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, prediction_input):
        logging.debug(prediction_input)
        if self.model is None:
            try:
                # load the model from a directory /folder given in the environment variables MODEL_REPO
                # assume the model file name as model.pkl
                model_repo = os.environ['MODEL_REPO']
                file_path = os.path.join(model_repo, "model.pkl")
                self.load_model(file_path)
            except KeyError:
                print("MODEL_REPO is undefined")
                # Otherwise, use the local model (in prediction-api folder)
                self.load_model('model.pkl')

        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        xNew = df[['ntp', 'pgc', 'dbp', 'tsft', 'si', 'bmi', 'dpf', 'age']]
        y_pred = self.model.predict(xNew)
        print(y_pred)
        logging.info(y_pred[0])
        status = (y_pred[0] > 0.5)
        print(status)
        return status

    def load_model(self, file_path):
        self.model = pickle.load(open(file_path, 'rb'))
