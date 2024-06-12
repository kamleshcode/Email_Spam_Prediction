# import joblib
# import numpy as np
# import pandas as pd
# from pathlib import Path

# class PredictionPipeline:
#     def __init__(self):
#         # Load the trained model and the CountVectorizer
#         self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
#         self.vectorizer = joblib.load(Path('artifacts/model_trainer/count_vectorizer.pkl'))
    
#     def predict(self, data):
#         # Ensure data is a string or a list of strings
#         if isinstance(data, str):
#             # If a single string is provided, convert it to a list with one element
#             data = [data]
#         elif not isinstance(data, list) or not all(isinstance(item, str) for item in data):
#             raise ValueError("Input data should be a string or a list of strings.")
        
#         # Transform the data using the CountVectorizer
#         transformed_data = self.vectorizer.transform(data)
        
#         # Make predictions using the transformed data
#         prediction = self.model.predict(transformed_data)

#         return prediction

# if __name__ == "__main__":
#     # Example usage
#     test_data = "hii"
    
#     prediction_pipeline = PredictionPipeline()
#     predictions = prediction_pipeline.predict(test_data)
#     print(f"Predictions: {predictions}")

import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.vectorizer = joblib.load(Path('artifacts/model_trainer/count_vectorizer.pkl'))
    
    
    def predict(self, data):
        transformed_data = self.vectorizer.transform(data)
        prediction = self.model.predict(transformed_data)

        return prediction
