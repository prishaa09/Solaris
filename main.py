import streamlit as st
import json
import requests
import base64
from PIL import Image
import io

# CONSTANTS
PREDICTED_LABELS = ['Normal', 'Solar Flare']
IMAGE_URL = "https://png.pngtree.com/thumb_back/fw800/background/20241126/pngtree-powerful-solar-flare-depiction-stunning-image-of-cosmic-energy-and-activity-image_16667986.jpg"
PREDICTED_LABELS.sort()

# Function to get prediction
def get_prediction(image_data):
    url = 'https://askai.aiclub.world/b5de9e88-76d5-4515-8b72-dafd9fa0e2ff'  # Replace with your AI endpoint URL
    r = requests.post(url, data=image_data)
    response = r.json()['predicted_label']
    score = r.json()['score']
    return response, score

# Set up the sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["About", "Prediction", "History"])

# About Page
if page == "About":
    st.title("About Solaris")
    st.image(IMAGE_URL, caption="Solar Flare Prediction")
    st.header("About the Web App")
    with st.expander("Web App 🌐"):
        st.subheader("Solar Flare Predictions")
        st.write("""
        My app is designed to predict and classify solar flare images into one of the following categories:
        1. Normal
        2. Solar Flare
        """)

# Prediction Page
elif page == "Prediction":
    st.title("Solar Flare Prediction")
    st.image(IMAGE_URL, caption="Upload your image for prediction")

    # File uploader for image
    image = st.file_uploader("Upload a solar flare image", type=['jpg', 'png', 'jpeg'])
    if image:
        # Convert the image to bytes
        img = Image.open(image).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        # Convert bytes to base64 encoding
        payload = base64.b64encode(byte_im)

        # File details
        file_details = {
            "file name": image.name,
            "file type": image.type,
            "file size": image.size
        }

        # Display the uploaded image
        st.image(img, caption="Uploaded Image")

        # Get prediction
        response, scores = get_prediction(payload)

        # Map response to labels
        response_label = PREDICTED_LABELS[response]

        # Display prediction results
        st.metric("Prediction Label", response_label)
        st.metric("Confidence Score", max(scores))

# History Page
elif page == "History":
    st.title("Prediction History")
    st.write("This page can be used to display past predictions.")
    st.write("Currently, no predictions are stored. You can implement a database or file storage to maintain history.")
