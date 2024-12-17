import streamlit as st
import pickle
import numpy as np
import sklearn

# Load the Naive Bayes model, vectorizer, and feature names
with open('naive_bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title("Spam Email Detection App")

# Pre-fill the text input with the sample instance
sample_text = (
    "Subject: You’ve Won a FREE Vacation! Claim Now! Body: Congratulations! 🎉 You have been randomly selected to win an all-expenses-paid trip to the Bahamas! 🏖️ This exclusive offer is only available for a limited time. To claim your free vacation, simply click the link below and complete the short survey. It takes less than 2 minutes!  Claim Your Vacation Now Hurry! This offer expires in 24 hours! Don't miss this once-in-a-lifetime opportunity to relax in paradise. 🌴Warm regards, The Free Vacation Team"
)

# Input field for the email text
email_text = st.text_area("Enter the email text:", value=sample_text, height=300)

# Predict button
if st.button("Predict"):
    # Transform the input text using the loaded vectorizer
    input_vector = vectorizer.transform([email_text])

    # Make prediction
    prediction = model.predict(input_vector)[0]

    # Display the result
    if prediction == 1:
        st.error("The email is predicted to be SPAM.")
    else:
        st.success("The email is predicted to be NOT SPAM.")
