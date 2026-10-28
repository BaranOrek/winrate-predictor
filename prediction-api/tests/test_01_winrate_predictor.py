import json
import os

import pandas as pd

# content of test_class.py
import winrate_predictor


class TestWinratePredictor:
    def test_predict_single_record(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_file = os.path.join(test_dir, "../testResources/prediction_request.json")
        test_model_file = os.path.join(test_dir, "../../models/model.pkl")
        with open(test_data_file) as json_file:
            data = json.load(json_file)
        dp = winrate_predictor.WinratePredictor()
        dp.load_model(test_model_file)
        status = dp.predict_single_record(data)
        assert status is not None