import streamlit as st
import joblib
from PIL import Image
import base64

# Load model
model = joblib.load("model.pkl")

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for a cleaner and modern look
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/soft-kill.png");
        font-family: 'Segoe UI', sans-serif;
    }
    .header {
        text-align: center;
        padding-bottom: 10px;
    }
    .result-box {
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Optional: Add a header image or logo
st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)

# Title & Intro
st.markdown('<div class="header"><h1>📰 Fake News Detector</h1></div>', unsafe_allow_html=True)
st.markdown("Enter the **news article text** below to check if it's Real or Fake. Our model uses **NLP** and **Machine Learning** to evaluate the content.")

# User Input
user_input = st.text_area("✍️ Paste or type the news article here:")

# Predict button
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        result = model.predict([user_input])[0]

        if result == 1:
            st.markdown('<div class="result-box" style="background-color: #d4edda; color: #155724;">✅ This article is likely REAL.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="background-color: #f8d7da; color: #721c24;">🚨 This article is likely FAKE.</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("📘 *Model trained using TF-IDF and Logistic Regression. Results may vary.*")
st.markdown("Created with ❤️ using Streamlit")
