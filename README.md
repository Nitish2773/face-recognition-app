# CertifySecure: Blockchain-Integrated Student Certificate Validation App with Flutter  
# Flask-Based Face Recognition  

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

## 🨠 Technical Architecture

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
### Dataset Creation and Training
The dataset training process involves capturing student facial images, preprocessing them, and training a Local Binary Pattern Histogram (LBPH) model for recognition.

#### 🔹 Step 1: Capture Training Data
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
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (200, 200))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"{dataset_path}/{timestamp}.jpg", face_resized)
                count += 1
```

### 💪 Training Requirements
- **Minimum Images:** 500 per student
- **Conditions:** Well-lit environment, various facial expressions
- **Recognition Confidence Threshold:** <100 (lower is better)

## 📊 Performance Metrics
| Metric                     | Value       |
|----------------------------|------------|
| Image Resolution           | 200x200 px |
| Training Dataset Size      | 500+ imgs  |
| Face Detection Accuracy    | >90%       |
| Recognition Accuracy       | 85-95%     |

## 🌐 API Endpoints
### Face Recognition Endpoint
```python
@app.route('/recognize', methods=['POST'])
def recognize_face():
    # Capture and process face
    # Compare with stored data
    # Verify against Firebase authentication records
    # Return authentication status
```

## 🌍 Deployment Options
- **Docker Deployment** for containerized execution
- **Google Cloud Run / AWS Lambda** for scalable hosting
- **Render** for easy deployment with auto-scaling

## 🌐 Setting Up Google Drive API
1. **Enable Google Drive API** in [Google Cloud Console](https://console.cloud.google.com/).
2. **Create a Service Account** and **Generate JSON Key**.
3. **Store `credentials.json` securely in the project.**

## 🔒 Obtaining Firebase Service Account JSON File
1. **Go to [Firebase Console](https://console.firebase.google.com/)**.
2. **Select Your Project** and navigate to **Service Accounts**.
3. **Generate a new private key** and download it.

## 📚 Future Enhancements
- **QR Code Verification** for instant certificate validation.
- **UI Enhancements** for better user experience.
- **Face Recognition with Deep Learning Models** for improved accuracy.

## 📞 Contact & Support
- **Email**: nitishkamisetti123@gmail.com
- **LinkedIn**: [Sri Nitish Kamisetti](https://www.linkedin.com/in/sri-nitish-kamisetti/)
- **GitHub**: [Nitish2773](https://github.com/Nitish2773)