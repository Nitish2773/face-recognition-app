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

## 🛠 Technical Architecture

### Project Structure
```
face-recognition-app/
│
├── app/
│   ├── main.py        # Face recognition processing
│   ├── train.py       # Model training script
│
├── utils/
│   ├── firebase_utils.py  # Firebase authentication integration
│   ├── drive_utils.py     # Google Drive storage integration
│
├── models/
│   ├── label.pkl   # Encoded label data
│   ├── face_recognition_model.pkl  # Trained face recognition model
│
├── .env              # Environment variables configuration
├── run.py            # Entry point to start Flask app
├── Dockerfile        # Docker container setup
├── requirements.txt  # Required dependencies
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
python run.py
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

### **Deploying on Render**

1. **Sign Up for Render** at [Render](https://render.com/).
2. **Create a New Web Service**.
3. **Connect GitHub Repository**.
4. **Select Environment**:
   - **Runtime**: Python 3.9+
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:5000 run:app`
5. **Add Environment Variables** (from `.env` file).
6. **Deploy & Monitor Logs**.

## 📝 Setting Up Google Drive API

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
5. **Download the JSON file** and store it in the project directory.

## 📄 Licensing
- Open-source under the **MIT License**

## 🤝 Contributing
1. **Fork the Repository**
2. **Create a New Branch** (`feature-new-feature`)
3. **Commit Changes**
4. **Push to Branch & Submit PR**

## 📞 Contact & Support
- **Email**: [support@Nitish](mailto:nitishkamisetti123@gmail.com)


