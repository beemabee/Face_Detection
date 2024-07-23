import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import os

# Fungsi untuk memuat gambar
def load_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image at {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image

# Fungsi untuk mendeteksi wajah
def detect_faces(model, image_path, target_size=(224, 224)):
    image = load_image(image_path, target_size)
    if image is None:
        return None, None
    original_image = cv2.imread(image_path)
    h, w, _ = original_image.shape
    image = np.expand_dims(image, axis=0)
    pred_bbox = model.predict(image)[0]
    pred_bbox = [int(pred_bbox[0] * w / target_size[0]), int(pred_bbox[1] * h / target_size[1]),
                 int(pred_bbox[2] * w / target_size[0]), int(pred_bbox[3] * h / target_size[1])]
    return original_image, pred_bbox

# Fungsi untuk menggambar bounding box
def draw_bounding_box(image, bbox):
    xmin, ymin, xmax, ymax = map(int, bbox)  # Pastikan koordinat adalah integer
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image

# Fungsi untuk mengunggah gambar
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image, pred_bbox = detect_faces(model, file_path)
        if original_image is not None and pred_bbox is not None:
            image_with_bbox = draw_bounding_box(original_image, pred_bbox)
            display_image(image_with_bbox)

# Fungsi untuk menampilkan gambar
def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel.config(image=image)
    panel.image = image

# Fungsi untuk memuat model terbaik
def load_best_model(model_path):
    if model_path.endswith('.h5') or model_path.endswith('.keras'):
        return load_model(model_path, custom_objects=get_custom_objects())
    else:
        raise ValueError("Unsupported model format. Please use a .h5 or .keras file.")

# Muat model terbaik
model_path = r'D:\Project\Face-Detection\model\tuned_keras.h5'
model = load_best_model(model_path)

# Buat jendela utama
root = tk.Tk()
root.title("Face Detection Application")

# Buat tombol untuk mengunggah gambar
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack()

# Buat panel untuk menampilkan gambar
panel = tk.Label(root)
panel.pack()

# Jalankan aplikasi
root.mainloop()