import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
@st.cache_resource
import gdown

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=14Mm9RcmzLlsn8mInk0pu3IUgoaONrnVu"  # Replace YOUR_FILE_ID
    output = "breast_cancer_model.keras"
    gdown.download(url, output, quiet=False)  # Download model from Google Drive
    model = tf.keras.models.load_model(output)
    return model


model = load_model()

# Define class labels
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

# Streamlit UI
st.title("Breast Cancer Detection from Ultrasound Images")
st.write("Upload an ultrasound image to classify it as Benign, Malignant, or Normal.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    image = image.resize((128, 128))  # Resize as per model input size
    image = np.array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Show prediction
    st.write(f"### Prediction: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}%")
