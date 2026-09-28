# 🐄 Livestock Breed Identification System

A full-stack AI/ML application for identifying livestock breeds from uploaded images.

The system uses **YOLOv8** to detect the main animal in an image and **MobileNetV2** to classify the detected animal into its breed. The application provides a web interface for uploading images and a FastAPI backend for performing predictions.

---

## 📌 Project Overview

Livestock breed identification can help in areas such as livestock management, agricultural research, and breed documentation.

This project provides an automated image-based approach:

**Input Image → Animal Detection → Image Cropping → Breed Classification → Prediction**

### Main Technologies

- **Python**
- **TensorFlow / Keras**
- **MobileNetV2**
- **YOLOv8**
- **FastAPI**
- **React + TypeScript**
- **Vite**
- **HTML / CSS**
- **OpenCV / PIL**
- **Uvicorn**

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     User / Browser  │
                    └──────────┬──────────┘
                               │
                               │ Upload Image
                               ▼
                    ┌─────────────────────┐
                    │ React + TypeScript  │
                    │      Frontend       │
                    └──────────┬──────────┘
                               │
                               │ HTTP Request
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       Backend       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      YOLOv8         │
                    │ Animal Detection    │
                    └──────────┬──────────┘
                               │
                         Animal Crop
                               │
                               ▼
                    ┌─────────────────────┐
                    │    MobileNetV2      │
                    │ Breed Classification│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Breed + Confidence  │
                    │      Prediction     │
                    └─────────────────────┘
```

---

## 🔄 How the System Works

### Step 1: Image Upload

The user uploads one or more livestock images through the web interface.

### Step 2: Animal Detection

The FastAPI backend receives the image and uses **YOLOv8** to detect the main animal.

The detected region is cropped from the original image.

### Step 3: Image Preprocessing

The cropped image is resized to:

```text
224 × 224 × 3
```

and prepared for the MobileNetV2 classifier.

### Step 4: Breed Classification

The processed image is passed to the trained **MobileNetV2** model.

The model produces probabilities for the supported livestock breeds.

### Step 5: Prediction

The API returns information such as:

- Predicted breed
- Confidence score
- Whether an animal was detected
- Detection confidence

---

## 🤖 Machine Learning Model

### Animal Detection

**YOLOv8** is used to identify and locate the main animal in the uploaded image.

### Breed Classification

The breed classifier is based on **MobileNetV2** with transfer learning.

The model uses:

- ImageNet-pretrained MobileNetV2
- `include_top=False`
- Global Average Pooling
- Dropout
- Softmax classification layer
- Adam optimizer
- Sparse categorical cross-entropy loss

The classifier accepts images of size:

```text
224 × 224 × 3
```

---

## 📂 Project Structure

```text
livestock-breed-id/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── inference.py
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── models/
│   │   └── breed_labels.json
│   │
│   ├── .env.example
│   ├── requirements.txt
│   └── yolov8n.pt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── styles.css
│   │
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── training/
│   └── train_classifier.py
│
├── README.md
├── start_project.bat
└── name of the animals.txt
```

---

## 📊 Dataset

The livestock breed image dataset used for training is **not included in this GitHub repository**.

### Why?

The dataset contains a large number of image files and would unnecessarily increase the size of the source-code repository.

The dataset was used locally during model training.

The repository contains the training script:

```text
training/train_classifier.py
```

This allows the training process and implementation to be reviewed without committing the complete image dataset.

---

## 🧠 Trained Model

The trained TensorFlow/Keras classifier:

```text
backend/models/breed_classifier.keras
```

is intentionally **not included in this repository**.

### Why?

The trained model is a binary model file and is kept separate from the source-code repository to keep the Git repository lightweight.

The repository does include:

```text
backend/models/breed_labels.json
```

which contains the breed-label mapping required by the application.

The model can be generated by running the training pipeline using the appropriate dataset.

---

## 🚀 Running the Backend

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install backend dependencies

```powershell
pip install -r backend/requirements.txt
```

### 3. Start FastAPI

From the `backend` directory:

```powershell
uvicorn app.main:app --reload --port 8000
```

The backend will run at:

```text
http://127.0.0.1:8000
```

### API documentation

FastAPI automatically provides interactive Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## 🌐 Running the Frontend

Open another terminal and navigate to:

```text
frontend
```

Install the dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173/
```

---

## 🔌 API Endpoints

### Health Check

```text
GET /health
```

Used to check whether the backend service is running.

### Breed Prediction

```text
POST /api/v1/predict
```

Accepts uploaded image files and returns the prediction results.

The API supports multiple image uploads.

---

## 📋 Example Prediction

Example response:

```json
{
  "predictions": [
    {
      "breed": "Bargur",
      "confidence": 0.8728,
      "animal_detected": true,
      "detection_confidence": 0.8227
    }
  ]
}
```

*Example values are provided to demonstrate the API response format.*

---

## 🎯 Key Features

- Image-based livestock breed identification
- YOLOv8-based animal detection
- MobileNetV2-based breed classification
- Transfer learning
- FastAPI REST API
- Interactive Swagger API documentation
- React + TypeScript frontend
- Multiple image upload support
- Prediction confidence scores
- Modular backend structure

---

## 🛠️ Training

The training code is available in:

```text
training/train_classifier.py
```

The training pipeline performs image loading, preprocessing, model creation, training, and saving of the trained classifier.

The trained model is saved locally as:

```text
backend/models/breed_classifier.keras
```

The dataset itself is intentionally excluded from GitHub because of its size.

---

## ⚠️ Repository Note

The following files/folders are intentionally excluded from Git:

```text
cattle/
animals/
.venv/
backend/models/breed_classifier.keras
```

These exclusions are defined in `.gitignore`.

This keeps the GitHub repository focused on the **source code, training implementation, configuration, and documentation**, rather than large datasets and generated model artifacts.

---

## 🎓 Project Purpose

This project was developed as an AI/ML full-stack project to demonstrate practical implementation of:

- Computer Vision
- Transfer Learning
- Image Classification
- Object Detection
- REST API development
- Frontend/backend integration
- Model inference

---

## 👨‍💻 Author

**Raushan Kumar**

MCA – Artificial Intelligence & Machine Learning

GitHub:

https://github.com/Raushankumar92
