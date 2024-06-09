import streamlit as st
import numpy as np
import subprocess
from src.emailproject.pipeline.prediction import PredictionPipeline

st.header("Email Spam Prediction", divider="rainbow")
st.write("Enter email and know whether it is spam or ham")

    
input = st.text_input("Enter Email Text", key="email")

def start_predicting():
    st.write("Magic is happening......")
    with st.spinner("Predicting"):
        data = np.array(input).reshape(1, -1)
        obj = PredictionPipeline()
        predicted_value = obj.predict(data)

        if predicted_value == 1:
            st.error("It is spam", icon="✅")
        elif predicted_value == 0:
            st.success("It is not spam", icon="🚨")

if st.button("Submit"):
    start_predicting()
    
