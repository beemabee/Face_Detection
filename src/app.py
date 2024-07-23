import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

@st.cache_resource
def load_face_model(model_path):
    return load_model(model_path)

def detect_face(image, face_model):
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    
    prediction = face_model.predict(image_expanded)[0]
    
    x_min, y_min, x_max, y_max = prediction
    x_min, x_max = int(x_min * w), int(x_max * w)
    y_min, y_max = int(y_min * h), int(y_max * h)
    
    return (x_min, y_min, x_max, y_max)

def draw_bounding_box(image, bbox):
    # Ensure we're working with a copy of the image
    image_with_box = image.copy()
    # Draw a thick red rectangle
    cv2.rectangle(image_with_box, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
    return image_with_box

# Load model
model_path = r'D:\Project\Face-Detection\model\tuned_keras.h5'
face_model = load_face_model(model_path)

st.title("Face Detection Application")
st.write("Upload an image to detect faces")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Detecting faces...")
    
    bbox = detect_face(image_array, face_model)
    
    # Draw bounding box
    image_with_box = draw_bounding_box(image_array, bbox)
    
    # Display image with bounding box
    st.image(image_with_box, caption='Detected Face.', use_column_width=True)
    st.write(f"Bounding box coordinates: {bbox}")