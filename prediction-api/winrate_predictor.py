
import json
import os

import pandas as pd
from flask import jsonify
import logging
from io import StringIO
import pickle


class WinratePredictor:
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
        xNew = df[['league', 'elo_home', 'elo_away', 'crowd_present',
                   'stadium_capacity', 'attendance_pct', 'travel_distance_km',
                   'timezones_crossed', 'is_derby', 'venue_familiarity']]
        y_pred = self.model.predict(xNew)
        print(y_pred)
        logging.info(y_pred[0])

        y_proba = self.model.predict_proba(xNew)

        result = {
            "prediction": str(y_pred[0]),
            "probabilities": {
                str(label): float(prob)
                for label, prob in zip(self.model.classes_, y_proba[0])
            }
        }
        print(result)
        return result

    def load_model(self, file_path):
        self.model = pickle.load(open(file_path, 'rb'))
