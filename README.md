# Face Detection

Proyek ini mengimplementasikan deteksi wajah menggunakan Convolutional Neural Network (CNN) dengan Keras dan transfer learning menggunakan MobileNet. Aplikasi ini memungkinkan pengguna untuk mengunggah gambar dan mendeteksi wajah di dalamnya.

## Persyaratan

Pastikan Anda memiliki perangkat lunak berikut terinstal di sistem Anda:

- Python min ver3.11
- pip (Python package installer)

## Langkah-langkah Instalasi

1. **Clone Repository**

   Clone repository ini ke lokal Anda menggunakan perintah berikut:

   ```bash
   git clone https://github.com/username/Face_Detection.git
   cd Face_Detection
   ```

2. **Install Dependencies**

   Install semua dependencies yang diperlukan menggunakan `pip`:

   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Aplikasi

1. **Jalankan Script Deteksi Wajah**

   Untuk menjalankan aplikasi deteksi wajah, gunakan perintah berikut:

   ```bash
   python src/face_detection.py
   ```

   Aplikasi ini akan membuka jendela GUI yang memungkinkan Anda untuk mengunggah gambar dan mendeteksi wajah di dalamnya.

## Struktur Proyek

Berikut adalah struktur direktori dari proyek ini:
```
Face_Detection/
│
├── model/
│ ├── development/
│ │ ├── MobileNet.ipynb
│ │ ├── Pre_trained_MobileNet.ipynb
│ │ ├── Testing.ipynb
│ │ └── CNN_Keras.ipynb
│ └── .gitattributes
│
├── dataset/
│ ├── images/
│ └── labels/
│
├── src/
│ └── face_detection.py
│
├── requirements.txt
└── README.md
```