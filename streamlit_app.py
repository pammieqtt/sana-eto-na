import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Inject custom CSS for light pink theme
st.markdown("""
    <style>
        .stApp {
            background-color: #ffe6f0;
        }
        .css-1d391kg, .css-1v0mbdj, .css-1dp5vir {  /* Sidebar background */
            background-color: #ffd6e8 !important;
        }
        .stButton>button {
            background-color: #ffb6c1;
            color: white;
            border: none;
            padding: 0.5em 1em;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #ff99b3;
        }
        h1, h2, h3, h4 {
            color: #cc3366;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained CNN model
model = load_model("cnn_model.h5")

# Main title
st.title("🐱🐶 Cat vs Dog Classifier")

# Sidebar project description
st.sidebar.header("📌 EMTECH FINAL PROJECT")
st.sidebar.markdown("""
This project is a deep learning-based image classifier that distinguishes between cats and dogs using a Convolutional Neural Network (CNN). It includes a training notebook and a Streamlit web app for easy image prediction.

**Features:**
- Upload an image of a cat or dog  
- Get instant prediction with confidence score  
- Try a sample image with one click
""")

# Image uploader
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    original_img = Image.open(uploaded_file)
    st.image(original_img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img = original_img.resize((64, 64)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    # Make prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    label = "Dog" if class_index == 1 else "Cat"
    confidence = prediction[0][class_index] * 100

    # Display results
    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence Score: {confidence:.2f}%")
