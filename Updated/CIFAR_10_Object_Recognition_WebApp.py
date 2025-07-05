import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mobilenetv2_cifar10_model.h5")
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image of size 96x96 or any small object image, and the model will predict the class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((96, 96))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🎯 Prediction: `{predicted_class}`")
    st.markdown(f"### 📊 Confidence: `{confidence*100:.2f}%`")
