import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def detect_face(model, image):
    h, w = image.shape[:2]
    processed_image = preprocess_image(Image.fromarray(image))
    pred_bbox = model.predict(processed_image)[0]
    
    xmin = int(pred_bbox[0] * w)
    ymin = int(pred_bbox[1] * h)
    xmax = int(pred_bbox[2] * w)
    ymax = int(pred_bbox[3] * h)
    
    return xmin, ymin, xmax, ymax

def draw_bounding_box(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image

model_path = r'D:\Project\Face-Detection\model\keras.h5'
model = load_model(model_path)

st.title("Face Detection Application")
st.write("Upload an image to detect faces")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Detecting faces...")
    
    image_array = np.array(image)
    
    bbox = detect_face(model, image_array)
    
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Draw bounding box
    image_with_box = draw_bounding_box(image_cv, bbox)
    
    # Convert back to RGB for display
    image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
    
    st.image(image_with_box_rgb, caption='Detected Face.', use_column_width=True)
    st.write(f"Bounding box coordinates: {bbox}")