Here's a rephrased and enhanced version of your Streamlit app with the project description added as a sidebar. I also polished the comments and structure for clarity and presentation:

```python
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

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
```

Let me know if you'd like to add a sample image button or enhance the layout with columns or tabs.
