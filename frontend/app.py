import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="AI Medical Diagnosis", layout="centered")

st.title("🧠 AI-Powered Medical Diagnosis")
st.write("Upload an image and select the diagnosis type:")

# Input
diagnosis_type = st.selectbox("Diagnosis Type", ["Pneumonia", "Brain Tumor"])
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔎 Diagnose"):
        with st.spinner("Sending to AI model..."):
            # Prepare request
            files = {"file": uploaded_file.getvalue()}
            diagnosis_key = "pneumonia" if diagnosis_type == "Pneumonia" else "tumor"
            api_url = f"http://127.0.0.1:8000/predict/{diagnosis_key}"

            try:
                response = requests.post(api_url, files=files)
                result = response.json()

                st.success(f"🩺 Prediction: **{result['prediction']}**")
                st.info(f"Confidence: **{result['confidence']}%**")

            except Exception as e:
                st.error("Error while connecting to API.")
                st.exception(e)
