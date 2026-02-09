# Automated Abdominal CT Scan Preprocessing and Disease Prediction

This repository contains the source code for my thesis project: an end-to-end web application designed to automate the preprocessing of abdominal CT scans and provide highly accurate disease predictions.

---

## 🚀 Research Highlights
* **Best Model:** Few-Shot Learning (Prototypical Networks).
* **Standard Accuracy:** **98.33%**
* **Comparative Analysis:** Performance compared against **MobileNetV2** and **ResNet50**.
* **Functionality:** Real-time image preprocessing (cleaning/denoising) followed by automated classification.

---

## 🛠️ Tech Stack
* **Backend:** Python / Flask
* **Frontend:** HTML5, CSS3, JavaScript
* **Deep Learning:** TensorFlow/Keras & PyTorch
* **Libraries:** OpenCV, NumPy, Matplotlib

---

## 💻 Features
* **Automated Preprocessing:** Automatically handles noise reduction, scaling, and resizing of raw CT scan uploads.
* **Multi-Model Support:** Includes logic for MobileNet and ResNet, though **Few-Shot Learning** is the primary engine.
* **User-Friendly Interface:** Simple drag-and-drop or upload functionality designed for medical professionals.
* **Instant Diagnosis:** Provides classification results and disease predictions immediately after processing.

---

## 📁 Project Structure
```text
├── models/             # (Local) Trained model weights (.keras, .h5, .pth)
├── static/             # CSS, JS, and UI assets (logo, styles, scripts)
├── templates/          # HTML pages (Index, Prediction Result)
├── uploads/            # Temporary storage for user-uploaded scans
├── uploads_cleaned/    # Storage for preprocessed images
├── app.py              # Main Flask application entry point
├── model_loader.py     # Logic for loading different model architectures
└── models.json         # Metadata and configuration for models
```
⚙️ Installation & Setup
Note: Due to file size constraints (GitHub's 100MB limit), the pre-trained weights in the models/ folder are not included in this repository.

Clone the repository:
```
Bash
git clone [https://github.com/ssijan/Automated-Abdominal-CT-Scan-Preprocessing-and-Disease-Prediction-Web-Application.git](https://github.com/ssijan/Automated-Abdominal-CT-Scan-Preprocessing-and-Disease-Prediction-Web-Application.git)
cd "Thesis Project"
Install Dependencies:

Bash
pip install -r requirements.txt
Add Model Weights: Place your protonet_best.pth, resnet50.h5, or .keras files inside the models/ directory.
```
Run the App:
```
Bash
python app.py
Access the app at http://127.0.0.1:5000
```
🎓 Thesis Credits
Author: Md. Sakibur Rahman, Jarin Tasmim Jinia
Topic: Explainable Deep Learning Framework for Multi-Class Classification of Abdominal Diseases in Computed Tomography Imaging
Year: 2026
