# import streamlit as st
# import pandas as pd
# from emailproject.pipeline.prediction import PredictionPipeline

# st.header("Email Spam Prediction", divider="rainbow")
# st.write("Enter email and know whether it is spam or ham")

# def start_predicting():
#     # st.title("Email Spam Prediction", divider="rainbow")
#     # st.write("Enter email and know whether it is spam or ham")
    
#     email_text = st.text_area("Enter Email Text")
    
#     if st.button("Predict"):
#         if email_text:
#             try:
#                 obj = PredictionPipeline()
#                 # Convert email_text to a list of one element (string)
#                 data = [email_text]
#                 predicted_value = obj.predict(data)
                
#                 if predicted_value[0] == 1:
#                     st.error("It is spam", icon="🚨")
#                 elif predicted_value[0] == 0:
#                     st.success("It is not spam", icon="✅")
#             except Exception as e:
#                 st.error(f"Error: {e}")

# if __name__ == "__main__":
#     start_predicting()

import streamlit as st
import pandas as pd
from emailproject.pipeline.prediction import PredictionPipeline

# Set page configuration
st.set_page_config(
    page_title="Email Spam Prediction",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

# Define a custom CSS style for the header
st.markdown(
    """
    <style>
    .stApp{
        background-image: url("static\image.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .header {
        text-align: center;
        color: #4CAF50;
        font-size: 3em;
        margin-bottom: 20px;
    }
    .subheader {
        filter: none;
        text-align: center;
        color: #F39C12;
        font-size: 1.5em;
        margin-bottom: 30px;
    }
    
    </style>
    """, unsafe_allow_html=True
)



st.header("Email Spam Prediction", divider="rainbow")
st.write("Enter email and know whether it is spam or ham")



# # Render the header and subheader
# st.markdown('<div class="header">Email Spam Prediction</div>', unsafe_allow_html=True)
# st.markdown('<div class="subheader">Enter the email text below to know if it is spam or not</div>', unsafe_allow_html=True)

# Function to start prediction
def start_predicting():
    email_text = st.text_area("Enter Email Text", height=200)
    
    if st.button("Predict", key="predict_button", help="Click to predict if the email is spam or not"):
        if email_text:
            with st.spinner("Analyzing..."):
                try:
                    obj = PredictionPipeline()
                    data = [email_text]
                    predicted_value = obj.predict(data)
                    
                    if predicted_value[0] == 1:
                        st.error("🚨 It is spam")
                    elif predicted_value[0] == 0:
                        st.success("✅ It is not spam")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some email text before predicting")

# Run the application
if __name__ == "__main__":
    start_predicting()

