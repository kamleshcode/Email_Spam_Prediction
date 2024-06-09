import os
import pandas as pd
from sklearn.metrics import accuracy_score
from emailproject.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from emailproject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        return accuracy

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        cv = joblib.load(os.path.join(self.config.root_dir, 'count_vectorizer.pkl'))

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column].values
        
        # for column in test_x.columns:
        #     if test_x[column].dtype == 'object':
        #         # Apply label encoding or any other encoding method
        #         le = LabelEncoder()
        #         test_x[column] = le.fit_transform(test_x[column].astype(str))
        test_x = cv.transform(test_x.iloc[:, 0])
        
        predicted_qualities = model.predict(test_x)

        accuracy = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as JSON
        scores = {"accuracy": accuracy}
        self.save_json(path=Path(self.config.metric_file_name), data=scores)
        