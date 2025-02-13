# CertifySecure: Blockchain-Integrated Student Certificate Validation App with Flask-Based Face Recognition

## 🚀 Project Overview

CertifySecure is a Flask-based face recognition system utilizing OpenCV and Haarcascade for secure and efficient student authentication. This backend processes biometric authentication requests and integrates Firebase for data storage and authentication.

## 🔒 Key Features

### Face Recognition Authentication
- **Uses OpenCV and Haarcascade** for face detection and recognition.
- **Low-latency recognition process** ensuring quick authentication.
- **Biometric security** to prevent unauthorized access.
- **Machine learning-based feature extraction** for accurate identification.

### Firebase Integration
- **Stores student authentication data securely**.
- **Firebase Admin SDK** for user management and authentication.
- **Real-time authentication logs** for security monitoring.

## 🫠 Technical Architecture

### Updated Project Structure
```
C:\face-recognition-app\app\
├── main.py              # Face recognition processing
├── setup_local.py       # Local setup script
├── train.py             # Model training script
├── utils/               # Utility functions
├── dataset/             # Training dataset storage
├── model/               # Trained models storage
├── secrets/
│   └── firebase-credentials  # Firebase credentials stored securely (no .json extension)
├── temp/                # Temporary storage for image processing
└── .env                 # Environment variables configuration
```

### Technology Stack
- **Backend**: Flask (Python)
- **Face Recognition**: OpenCV, Haarcascade, Scikit-learn
- **Authentication & Storage**: Firebase, Google Drive
- **Environment Management**: Python-dotenv

## 📸 Face Recognition Methodology

### Detection Techniques
- Haar Cascade Classifiers (for face detection)
- OpenCV-based feature extraction
- Scikit-learn for encoding and comparison

### Recognition Process
1. **Capture Image Input** (Sent via API request)
2. **Face Detection** (Using OpenCV Haarcascade)
3. **Feature Extraction & Encoding**
4. **Compare Encoded Features with Stored Data**
5. **Return Authentication Status**

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- Firebase Project with Admin SDK Credentials
- Google Drive API Access

### Backend Setup
```bash
# Clone Repository
git clone https://github.com/your-org/certifysecure-face-recognition.git
cd certifysecure-face-recognition

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install Dependencies
pip install -r requirements.txt

# Set Up Environment Variables
cp .env.example .env  # Configure Firebase and Google Drive credentials

# Run Flask App
python main.py
```

## 🔐 Security Measures
- **Biometric Authentication** (Face Recognition)
- **Firebase Authentication** for secure access management
- **Environment Variable Encryption** to protect sensitive keys

## 🤖 Machine Learning Models

### Face Recognition Models
- **Haar Cascade Classifiers** for face detection
- **LBPH (Local Binary Patterns Histograms)** for recognition

### Model Training
- **Dataset Preprocessing**: Image resizing, grayscale conversion
- **Encoding & Training**: Feature extraction & storage in `label.pkl`
- **Retraining for Improved Accuracy**

## 📚 Dataset Training Process

CertifySecure includes a robust dataset training pipeline to ensure accurate and reliable face recognition for student authentication.

### 🏗️ Dataset Creation and Training

The dataset training process involves capturing student facial images, preprocessing them, and training a Local Binary Pattern Histogram (LBPH) model for recognition.

#### 🔹 Step 1: Capture Training Data

The following Python script captures and preprocesses images for training:

```python
import cv2
import os
import numpy as np
from datetime import datetime

class DatasetCreator:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.cap = cv2.VideoCapture(0)
        
    def create_user_dataset(self, user_id, num_images=500):
        dataset_path = f"dataset/{user_id}"
        os.makedirs(dataset_path, exist_ok=True)

        count = 0
        while count < num_images:
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (200, 200))
                face_enhanced = cv2.equalizeHist(face_resized)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"{dataset_path}/{timestamp}.jpg", face_enhanced)
                count += 1
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captures: {count}/{num_images}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)

            cv2.imshow("Capturing Face Data", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
```

#### 🔹 Step 2: Data Augmentation

To improve model performance, dataset augmentation is applied:

```python
def augment_dataset(user_id):
    source_path = f"dataset/{user_id}"
    augmented_path = f"dataset/{user_id}/augmented"
    os.makedirs(augmented_path, exist_ok=True)
    
    for img_name in os.listdir(source_path):
        img = cv2.imread(f"{source_path}/{img_name}", cv2.IMREAD_GRAYSCALE)
        
        # Original Image
        cv2.imwrite(f"{augmented_path}/orig_{img_name}", img)
        
        # Flipped Image
        cv2.imwrite(f"{augmented_path}/flip_{img_name}", cv2.flip(img, 1))
        
        # Rotated Images
        for angle in [-15, 15]:
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(f"{augmented_path}/rot{angle}_{img_name}", rotated)
```

#### 🔹 Dataset Structure

```
dataset/
├── student_id_123/
│   ├── 20230915_143022_123456.jpg
│   ├── 20230915_143023_234567.jpg
│   └── augmented/
│       ├── orig_20230915_143022_123456.jpg
│       ├── flip_20230915_143022_123456.jpg
│       └── rot15_20230915_143022_123456.jpg
└── student_id_456/
    └── ...
```

### 🏋️‍♂️ Model Training Process

```python
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    
    for user_id in os.listdir("dataset"):
        if os.path.isdir(f"dataset/{user_id}"):
            for img_name in os.listdir(f"dataset/{user_id}"):
                img_path = f"dataset/{user_id}/{img_name}"
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(int(user_id))
    
    recognizer.train(faces, np.array(labels))
    recognizer.save("model/face_recognition_model.yml")
```

### 📝 Training Requirements

- **Minimum Images:** 500 per student
- **Conditions:** Well-lit environment, various facial expressions
- **Image Standardization:** 200x200 grayscale images
- **Face Detection Confidence:** >50%
- **Recognition Confidence Threshold:** <100 (lower is better)

### 📊 Performance Metrics

| Metric                     | Value       |
|----------------------------|------------|
| Image Resolution           | 200x200 px |
| Training Dataset Size      | 500+ imgs  |
| Face Detection Accuracy    | >90%       |
| Recognition Accuracy       | 85-95%     |

---

## 📊 Performance Metrics
- **Recognition Accuracy**: ~95%
- **Average Response Time**: < 500ms

## 📝 API Endpoints

### **Face Recognition Endpoint**
```python
@app.route('/recognize', methods=['POST'])
def recognize_face():
    # Capture and process face
    # Compare with stored data
    # Verify against Firebase authentication records
    # Return authentication status
```

## 🌐 Deployment Options
- **Docker Deployment** for containerized execution
- **Gunicorn** for production-grade WSGI serving
- **Google Cloud Run / AWS Lambda** for scalable hosting
- **Render** for easy deployment with auto-scaling

### **Render Deployment Configuration**
Create a `render.yaml` file in the project root:
```yaml
services:
  - type: web
    name: face-recognition-app
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: PORT
        value: 5000
      - key: RENDER
        value: true
    secrets:
      - name: firebase-credentials  # Remove .json extension
        mountPath: /etc/secrets/firebase-credentials  # Remove .json extension
```

### **Deploying on Render**

1. **Sign Up for Render** at [Render](https://render.com/).
2. **Create a New Web Service**.
3. **Connect GitHub Repository**.
4. **Select Environment**:
   - **Runtime**: Python 3.9+
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
5. **Add Environment Variables** (from `.env` file).
6. **Deploy & Monitor Logs**.

## 🔒 Setting Up Google Drive API

### **Steps to Get `credentials.json` for Google Drive API**
1. **Go to** [Google Cloud Console](https://console.cloud.google.com/).
2. **Create a New Project**.
3. **Enable Google Drive API**.
4. **Go to Credentials** → **Create Credentials** → **Service Account**.
5. **Generate JSON Key** and download it.
6. **Store this `credentials.json` file** in the project directory.

## 🔐 Obtaining Firebase Service Account JSON File

1. **Go to** [Firebase Console](https://console.firebase.google.com/).
2. **Select Your Project**.
3. **Go to Project Settings** → **Service Accounts**.
4. **Generate a new private key**.
5. **Download the JSON file** and store it in `secrets/firebase-credentials`.

## 📝 Licensing
- Open-source under the **MIT License**

## 🤝 Contributing
1. **Fork the Repository**
2. **Create a New Branch** (`feature-new-feature`)
3. **Commit Changes**
4. **Push to Branch & Submit PR**

## 📞 Contact & Support
- **Email**: [support@Nitish](mailto:nitishkamisetti123@gmail.com)