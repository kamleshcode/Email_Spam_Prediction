import pandas as pd
import os
from emailproject import logger
from sklearn.naive_bayes import MultinomialNB
import joblib
from emailproject.entity.config_entity import ModelTrainerConfig
from sklearn.feature_extraction.text import CountVectorizer


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column].values
        test_y = test_data[self.config.target_column].values
        
        cv = CountVectorizer()
        train_x = cv.fit_transform(train_x.iloc[:, 0])  # Assuming the text data is in the first column
        test_x = cv.transform(test_x.iloc[:, 0])

        nb = MultinomialNB()
        nb.fit(train_x, train_y.ravel())

        joblib.dump(nb, os.path.join(self.config.root_dir, self.config.model_name))