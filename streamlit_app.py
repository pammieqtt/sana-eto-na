import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained CNN model
model = load_model("cnn_model.h5")

# App title
st.title("🐱🐶 Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load original image for display
    original_img = Image.open(uploaded_file)
    st.image(original_img, caption="Uploaded Image", use_column_width=True)

    # Resize a copy for prediction
    img = original_img.resize((64, 64)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    label = "Dog" if class_index == 1 else "Cat"
    confidence = prediction[0][class_index] * 100

    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.2f}%")
