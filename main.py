import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import pickle
from xgboost import XGBClassifier
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Set the page layout
# st.set_page_config(page_title="Sidebar Navigation", layout="wide")

# constants
IMG_SIZE = (224, 224)
IMG_ADDRESS = "https://svs.gsfc.nasa.gov/vis/a010000/a011100/a011168/Raining_Loops_Still_2.jpg"
IMAGE_NAME = "user_image.png"
CLASS_LABEL = ["Normal", "SolarFlare"]
CLASS_LABEL.sort()

IMAGE_URL = "https://www.science.org.au/curious/sites/default/files/article-banner-image/flickr-gsfc-15403844862-v2.jpg"

@st.cache_resource
def get_ConvNeXtXLarge_model():

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.ConvNeXtXLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["About",  "Classification", "History"]
)

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Main Content Area
if page == "About":
    st.image(IMAGE_URL, caption = "SolarFlare Image Classification")
    st.title("Welcome to the About Page")
    st.header("About the Web App")
    with st.expander("Web App 🌐"):
        st.subheader("Solar Flare Predictions")
        st.write("""
        My app is designed to predict and classify solar flare images into one of the following categories:
        1. Normal
        2. Solar Flare
        """)

elif page == "Classification":
    # get the featurization model
    ConvNeXtXLarge_featurized_model = get_ConvNeXtXLarge_model()
    # load ultrasound image
    classification_model = load_sklearn_models("xgb_best_model.pkl")


    # web app

    # title
    st.title("SolarFlare Image Classification")
    # image
    st.image(IMG_ADDRESS, caption = "SolarFlare Image Classification")

    # input image
    st.subheader("Please Upload a SolarFlare image")

    # file uploader
    image = st.file_uploader("Please Upload a SolarFlare Image", type = ["jpg", "png", "jpeg"], accept_multiple_files = False, help = "Upload an Image")

    if image:
        user_image = Image.open(image)
        # save the image to set the path
        user_image.save(IMAGE_NAME)
        # set the user image
        st.image(user_image, caption = "User Uploaded Image")

        #get the features
        with st.spinner("Processing......."):
            image_features = featurization(IMAGE_NAME, ConvNeXtXLarge_featurized_model)
            model_predict = classification_model.predict(image_features)
            result_label = CLASS_LABEL[model_predict[0]]
            st.success(f"Prediction: {result_label}")

        # Save the result in history
        st.session_state["history"].append({"name": image.name, "label": result_label, "image": user_image})

elif page == "History":
    st.title("Classification History")

    if st.session_state["history"]:
        for entry in st.session_state["history"]:
            with st.expander(f"{entry['name']} - {entry['label']}"):
                st.image(entry["image"], caption=f"Prediction: {entry['label']}")
    else:
        st.write("No history available.")