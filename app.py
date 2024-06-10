import streamlit as st
import pandas as pd
from emailproject.pipeline.prediction import PredictionPipeline

def start_predicting():
    st.title("Email Spam Prediction")
    st.write("Enter email and know whether it is spam or ham")
    
    email_text = st.text_area("Enter Email Text")
    
    if st.button("Predict"):
        if email_text:
            try:
                obj = PredictionPipeline()
                # Convert email_text to a list of one element (string)
                data = [email_text]
                predicted_value = obj.predict(data)
                
                if predicted_value[0] == 1:
                    st.error("It is spam", icon="🚨")
                elif predicted_value[0] == 0:
                    st.success("It is not spam", icon="✅")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    start_predicting()
