import streamlit as st
import json
import requests
import base64
from PIL import Image
import io

#CONSTANTS
PREDICTED_LABELS = ['Normal', 'Solar Flare']
IMAGE_URL = "https://png.pngtree.com/thumb_back/fw800/background/20241126/pngtree-powerful-solar-flare-depiction-stunning-image-of-cosmic-energy-and-activity-image_16667986.jpg"

PREDICTED_LABELS.sort()

def get_prediction(image_data):
  #replace your image classification ai service endpoint URL
  url = 'https://askai.aiclub.world/b5de9e88-76d5-4515-8b72-dafd9fa0e2ff'
  r = requests.post(url, data=image_data)
  response = r.json()['predicted_label']
  score = r.json()['score'][response]
  #print("Predicted_label: {} and confidence_score: {}".format(response,score))
  return response, score



#Building the website

#title of the web page
st.title("Solaris")

#setting the main picture
st.image(IMAGE_URL, caption = "Image Classification")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.subheader("Solar Flare Predictions")
    st.write("""My app is designed to predict and classify solar flare images into one of the following categories :
    1. Normal
    2. Solar Flare""")

#setting file uploader
image =st.file_uploader("Upload a solar flare image",type = ['jpg','png','jpeg'])
if image:
  #converting the image to bytes
  img = Image.open(image).convert('RGB') #ensuring to convert into RGB as model expects the image to be in 3 channel
  buf = io.BytesIO()
  img.save(buf,format = 'JPEG')
  byte_im = buf.getvalue()

  #converting bytes to b64encoding
  payload = base64.b64encode(byte_im)

  #file details
  file_details = {
    "file name": image.name,
    "file type": image.type,
    "file size": image.size
  }

  #write file details
  #st.write(file_details) #uncomment if you need to show file details

  #setting up the image
  st.image(img)

  #predictions
  response, scores = get_prediction(payload)

  #if you are using the model deployment in navigator
  #you need to define the labels
  response_label = PREDICTED_LABELS[response]

  st.metric("Prediction Label",response_label)
  st.metric("Confidence Score", max(scores))
