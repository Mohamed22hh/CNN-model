import os
import json
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import plotly.express as px
import gdown

# Set the page configuration for better UI
st.set_page_config(
    page_title='Plant Disease Classifier',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Working directory setup
working_dir = os.path.dirname(os.path.abspath(__file__))

# Google Drive model file ID (replace with your model's drive ID)
GDRIVE_FILE_ID = '1pexWEDB_n8NzmKQkz16-Ktd5YDH2i3Ag'

# Function to download the model from Google Drive if not already present
@st.cache_resource
def download_model():
    model_path = os.path.join(working_dir, 'plant_disease_prediction_model.h5')
    if not os.path.exists(model_path):
        with st.spinner('Downloading model...'):
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, model_path, quiet=False)
    return model_path

# Function to load the model
@st.cache_resource
def load_model():
    model_path = download_model()
    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()

# Load the class names
class_indices_path = os.path.join(working_dir, "class_indices.json")
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Convert class_indices keys to integers for indexing
index_to_class = {int(k): v for k, v in class_indices.items()}

# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = index_to_class[predicted_class_index]
    return predicted_class_name, predictions

# Apply custom CSS styling
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Custom CSS file not found at {file_name}. Skipping custom styles.")

# Load custom CSS
css_file_path = os.path.join(working_dir, "styles.css")
local_css(css_file_path)

# Sidebar content with custom styling
st.sidebar.markdown("<h1 style='text-align: center; color: white;'>🌿 Plant Disease Classifier</h1>", unsafe_allow_html=True)
st.sidebar.info("""
This application uses a deep learning model to classify plant diseases from images of plant leaves.
""")
st.sidebar.write("Upload an image of a plant leaf, and click **Classify** to predict the disease.")

# Main content with tabs
tab1, tab2 = st.tabs(['🖼️ Classification', 'ℹ️ About'])

with tab1:
    st.markdown("<h2 style='text-align: center;'>Plant Disease Classification</h2>", unsafe_allow_html=True)
    st.write("")

    # Use columns to center the upload button
    col_space1, col_upload, col_space2 = st.columns([1, 2, 1])

    with col_upload:
        uploaded_image = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Upload an image of a plant leaf."
        )

    if uploaded_image is not None:
        # Display uploaded image with specified width
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', width=300)  # Adjust the width as needed

        # Add a button with custom styling
        classify_button = st.button('🌱 Classify Image', key='classify_button')

        if classify_button:
            with st.spinner('Classifying...'):
                prediction, probabilities = predict_image_class(model, uploaded_image)
            st.success(f'**Prediction:** {prediction}')

            # Display probabilities as an interactive plotly chart
            st.subheader('Prediction Probabilities')
            prob_df = pd.DataFrame({
                'Disease': [index_to_class[i] for i in range(len(probabilities))],
                'Probability': probabilities * 100
            })

            fig = px.bar(
                prob_df,
                x='Disease',
                y='Probability',
                color='Probability',
                labels={'Probability': 'Probability (%)'},
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)

            # Expanded disease information
            st.subheader('Disease Information')
            disease_info = {
                'Healthy': 'The plant is healthy. No treatment is needed.',
                'Bacterial Blight': 'A bacterial disease causing lesions on leaves. Treatment includes copper-based sprays and removing infected plant parts.',
                'Leaf Rust': 'A fungal disease causing orange-brown pustules. Treatment involves fungicides and improving air circulation.',
                'Powdery Mildew': 'A fungal disease with white powdery spots. Treatment includes sulfur sprays and reducing humidity.',
            }
            st.write(disease_info.get(prediction, "Detailed information not available."))

    else:
        st.info('Please upload an image to classify.')

with tab2:
    st.markdown("<h2 style='text-align: center;'>About this App</h2>", unsafe_allow_html=True)
    st.write("""
    **Plant Disease Classifier** is a deep learning-powered application that identifies diseases in plants from leaf images. It helps farmers and gardeners quickly diagnose plant health issues.

    **How to use:**
    1. Go to the **Classification** tab.
    2. Upload a clear image of a plant leaf.
    3. Click **Classify Image** to see the prediction.

    **Model Information:**
    - Trained on a comprehensive dataset of plant leaf images
    - Uses state-of-the-art convolutional neural network architecture
    - Supports multiple plant disease classifications

    **Disclaimer:** This tool is for informational purposes and should not replace professional agricultural advice.
    """)

# Footer with custom styling
st.markdown("""
<style>
footer {
    visibility: hidden;
}
</style>
<div style='text-align: center; padding: 10px;'>
    <hr>
    © 2024 Plant Disease Classifier • <a href='mailto:contact@plantdiseaseapp.com'>Contact</a>
</div>
""", unsafe_allow_html=True)
