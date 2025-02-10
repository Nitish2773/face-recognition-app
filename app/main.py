from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import os
import logging
import base64
import tempfile
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = Flask(__name__)
CORS(app)
ef initialize_firebase_credentials():
    """
    Initialize Firebase credentials from environment variable
    """
    try:
        # Get Firebase credentials path from environment variable
        firebase_cred_path = os.getenv('FIREBASE_CRED_PATH')
        
        if not firebase_cred_path:
            logging.error("FIREBASE_CRED_PATH environment variable not set")
            return None
        
        # Check if file exists
        if not os.path.exists(firebase_cred_path):
            logging.error(f"Firebase credentials file not found: {firebase_cred_path}")
            return None
        
        logging.info(f"Firebase credentials loaded from: {firebase_cred_path}")
        return firebase_cred_path
    
    except Exception as e:
        logging.error(f"Firebase credentials initialization error: {e}")
        return None
class FaceRecognizer:
    def __init__(self, firebase_cred_path):
        # Initialize Firebase
        try:
            # Log the exact path being used
            logging.info(f"Attempting to initialize Firebase with: {firebase_cred_path}")
            
            # Initialize Firebase
            cred = credentials.Certificate(firebase_cred_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            logging.info("Firebase initialized successfully")
        except Exception as e:
            logging.error(f"Firebase initialization failed: {e}")
            raise

        # Model and detection configurations
        self.MODEL_PATH = os.getenv('MODEL_PATH', "model/face_recognition_model.pkl")
        self.LABEL_DICT_PATH = os.getenv('LABEL_DICT_PATH', "model/label_dict.pkl")
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        self.load_recognizer()

   def load_recognizer(self):
        """Load pre-trained face recognition model"""
        try:
            # Verify model paths exist
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {self.MODEL_PATH}")
            
            if not os.path.exists(self.LABEL_DICT_PATH):
                raise FileNotFoundError(f"Label dictionary not found: {self.LABEL_DICT_PATH}")
            
            # Load scikit-learn model
            with open(self.MODEL_PATH, "rb") as f:
                self.face_recognizer = pickle.load(f)
            
            # Load label dictionary
            with open(self.LABEL_DICT_PATH, "rb") as f:
                self.label_dict = pickle.load(f)
            
            logging.info("Model and label dictionary loaded successfully")
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            raise


    def preprocess_image(self, image):
        """Preprocess and extract face from image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance image quality
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                logging.warning("No faces detected")
                return None
            
            # Take first detected face
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            
            # Resize for recognition
            face = cv2.resize(face, (200, 200))
            return face
        
        except Exception as e:
            logging.error(f"Image preprocessing error: {e}")
            return None

    def recognize_face(self, image):
        """Recognize face and retrieve user information"""
        try:
            # Preprocess image
            face = self.preprocess_image(image)
            
            if face is None:
                return None, None
            
            # Flatten the face for prediction
            face_flattened = face.flatten()
            
            # Predict using scikit-learn KNN
            try:
                prediction = self.face_recognizer.predict([face_flattened])
                label = prediction[0]
                
                # Find the corresponding user ID
                for user_id, user_info in self.label_dict.items():
                    if user_info['index'] == label:
                        # Additional verification from Firestore
                        user_ref = self.db.collection('users').document(user_id)
                        user_doc = user_ref.get()
                        
                        if user_doc.exists:
                            user_data = user_doc.to_dict()
                            
                            # Log recognition event
                            self.log_recognition_event(user_id)
                            
                            return {
                                'id': user_id,
                                'name': user_data.get('name', 'Unknown'),
                                'role': user_data.get('role', 'Unknown')
                            }, 1.0  # Confidence for KNN is typically 1.0
                
                # If no user found
                return None, None
            
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                return None, None
        
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            return None, None

    def log_recognition_event(self, user_id):
        """Log recognition event to Firestore"""
        try:
            events_ref = self.db.collection('recognition_events')
            events_ref.add({
                'user_id': user_id,
                'timestamp': datetime.now(),
                'method': 'face_recognition'
            })
        except Exception as e:
            logging.error(f"Event logging error: {e}")

# Flask Routes
def create_recognizer():
    """Create recognizer with error handling"""
    try:
        # Get Firebase credentials path
        firebase_cred_path = initialize_firebase_credentials()
        
        if not firebase_cred_path:
            logging.critical("Failed to obtain Firebase credentials")
            return None
        
        # Create and return recognizer
        recognizer = FaceRecognizer(firebase_cred_path)
        return recognizer
    
    except Exception as e:
        logging.critical(f"Failed to initialize recognizer: {e}")
        return None

# Initialize recognizer
recognizer = create_recognizer()

@app.route('/recognize', methods=['POST'])
def api_recognize():
    # Check if recognizer is initialized
    if recognizer is None:
        return jsonify({
            "success": False,
            "message": "Face recognition service not available"
        }), 500

    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "No image uploaded"
            }), 400

        # Save uploaded image
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Recognize
        person_info, confidence = recognizer.recognize_face(image)

        if person_info:
            return jsonify({
                "success": True,
                "user_id": person_info['id'],
                "name": person_info.get('name', 'Unknown'),
                "role": person_info['role'],
                "confidence": float(confidence)
            })
        else:
            return jsonify({
                "success": False,
                "message": "Face not recognized"
            }), 404

    except Exception as e:
        logging.error(f"API recognition error: {e}")
        return jsonify({
            "success": False,
            "message": "Internal server error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check if recognizer is initialized
        if recognizer is None:
            return jsonify({
                "success": False,
                "message": "Recognizer not initialized"
            }), 500
        
        # Check Firebase connection
        users_ref = recognizer.db.collection('users')
        users = users_ref.limit(1).get()
        
        return jsonify({
            "success": True,
            "message": "Service is healthy",
            "recognizer_loaded": True,
            "firebase_connected": True
        }), 200
    
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            "success": False,
            "message": f"Service is not healthy: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Additional error handling for startup
    if recognizer is None:
        logging.critical("Failed to start server due to recognizer initialization failure")
        exit(1)
    
    # Use PORT environment variable or default to 5000
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)