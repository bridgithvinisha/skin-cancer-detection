import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils.predict import predict_image

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🧠 Skin Cancer Detection System")

# 🔥 Important warning
st.warning("⚠️ This AI model is for educational purposes only and not a medical diagnosis.")

st.write("Upload a dermoscopic image to classify skin lesion")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            label, confidence = predict_image(img)

        st.success(f"Prediction: {label.upper()}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        # Risk logic
        if label == "mel":
            st.error("⚠️ High Risk: Possible Melanoma")
        else:
            st.success("✅ Lower Risk")

st.markdown("---")
st.caption("Developed using Deep Learning & Streamlit")