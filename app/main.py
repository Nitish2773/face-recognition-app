from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pickle
import logging
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class FaceRecognizer:
    def __init__(self):
        try:
            # Initialize face cascade with explicit error checking
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            logger.info(f"Loading cascade classifier from: {cascade_path}")
            
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Cascade file not found at: {cascade_path}")
                
            self.face_cascade = cv2.CascadeClassifier()
            success = self.face_cascade.load(cascade_path)
            
            if not success:
                raise RuntimeError("Failed to load cascade classifier")
                
            logger.info("Face cascade classifier loaded successfully")
            
            # Get the current working directory
            current_dir = os.getcwd()
            
            # Check if running on Render
            if os.getenv('RENDER') == 'true':
                # Use Render's secret file path
                cred_path = '/etc/secrets/firebase-credentials'
                logger.info("Running on Render environment")
            else:
                # Local development path
                cred_path = os.path.join(current_dir, 'secrets', 'firebase-credentials.json')
                logger.info("Running on local environment")

            logger.info(f"Looking for Firebase credentials at: {cred_path}")
            
            # Create secrets directory if it doesn't exist locally
            if not os.getenv('RENDER') == 'true' and not os.path.exists('secrets'):
                os.makedirs('secrets')
                logger.info("Created secrets directory")

            if not os.path.exists(cred_path):
                raise FileNotFoundError(f"Firebase credentials not found at {cred_path}")

            cred = credentials.Certificate(cred_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info("Firebase initialized successfully")

            # Get base directory
            base_dir = os.path.abspath(os.path.dirname(__file__))
            
            # Set model paths
            self.MODEL_PATH = os.path.join(base_dir, "model", "face_recognition_model.yml")
            self.LABEL_DICT_PATH = os.path.join(base_dir, "model", "label_dict.pkl")
            
            logger.info(f"\n=== Model Paths ===")
            logger.info(f"Base directory: {base_dir}")
            logger.info(f"Model path: {self.MODEL_PATH}")
            logger.info(f"Label dict path: {self.LABEL_DICT_PATH}")
            
            # Verify model directory exists
            model_dir = os.path.join(base_dir, "model")
            if not os.path.exists(model_dir):
                logger.info(f"Model directory not found. Creating at: {model_dir}")
                os.makedirs(model_dir)
            
            # Verify model files exist
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at: {self.MODEL_PATH}")
            if not os.path.exists(self.LABEL_DICT_PATH):
                raise FileNotFoundError(f"Label dictionary not found at: {self.LABEL_DICT_PATH}")
            
            # Load recognizer
            self.load_recognizer()
            logger.info("Model initialization successful")
            
        except Exception as e:
            logger.error(f"\n=== Initialization Error ===")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error("Directory contents:")
            for root, dirs, files in os.walk(os.getcwd()):
                level = root.replace(os.getcwd(), '').count(os.sep)
                indent = ' ' * 4 * level
                logger.error(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    logger.error(f"{subindent}{f}")
            logger.error(traceback.format_exc())
            raise

    def verify_cascade(self):
        """Verify that the cascade classifier is properly loaded"""
        if self.face_cascade is None or self.face_cascade.empty():
            raise RuntimeError("Face cascade classifier is not properly initialized")

    def load_recognizer(self):
        """Load the face recognizer model and label dictionary"""
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.read(self.MODEL_PATH)
            with open(self.LABEL_DICT_PATH, "rb") as f:
                self.label_dict = pickle.load(f)
            logger.info("Model and label dictionary loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or label dictionary: {e}")
            raise

    def preprocess_image(self, image_path):
        """Preprocess the input image for face recognition"""
        try:
            # Verify cascade classifier
            self.verify_cascade()
            
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return None

            # Verify image dimensions
            if img.shape[0] == 0 or img.shape[1] == 0:
                logger.error("Invalid image dimensions")
                return None

            # Apply histogram equalization
            img = cv2.equalizeHist(img)
            
            logger.info(f"Processing image of size: {img.shape}")
            
            # Detect faces with explicit error checking
            if self.face_cascade is None:
                raise RuntimeError("Face cascade is None")
                
            faces_detected = self.face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces_detected) == 0:
                logger.info("No faces detected in image")
                return None

            # Process the first detected face
            x, y, w, h = faces_detected[0]
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                logger.error("Invalid face coordinates detected")
                return None

            face = cv2.resize(img[y:y+h, x:x+w], (200, 200))
            return face

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def recognize_face(self, image_path):
        """Recognize a face in the given image"""
        face = self.preprocess_image(image_path)
        if face is None:
            return None, None

        try:
            label, confidence = self.face_recognizer.predict(face)
            person_info = self.label_dict.get(label, {"id": "Unknown", "role": "Unknown"})
            
            logger.info(f"\n=== Recognition Attempt ===")
            logger.info(f"ID: {person_info['id']}")
            logger.info(f"Confidence: {confidence}")
            
            user_ref = self.db.collection('users').document(person_info['id'])
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                person_info['role'] = user_data.get('role', 'Unknown')
                
                if confidence > 200:  # Adjust this threshold as needed
                    logger.info("Recognition Failed: Confidence too low")
                    return None, None
                    
                logger.info(f"Recognition Successful!")
                logger.info(f"Role: {person_info['role']}")
                
                # Log the recognition event
                self.log_recognition_event(person_info['id'], confidence)
                
                return person_info, confidence
            else:
                logger.error(f"User {person_info['id']} not found in Firebase")
                return None, None

        except Exception as e:
            logger.error(f"Error during face recognition: {e}")
            logger.error(traceback.format_exc())
            return None, None

    def log_recognition_event(self, user_id, confidence):
        """Log face recognition events to Firebase"""
        try:
            self.db.collection('recognition_logs').add({
                'user_id': user_id,
                'confidence': confidence,
                'timestamp': datetime.now(),
            })
        except Exception as e:
            logger.error(f"Error logging recognition event: {e}")

# Initialize the face recognizer
recognizer = FaceRecognizer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/recognize', methods=['POST'])
def api_recognize():
    """Face recognition endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "No image file provided"
            }), 400

        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({
                "success": False,
                "message": "Empty filename"
            }), 400

        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        # Save and process image
        image_path = os.path.join('temp', image_file.filename)
        image_file.save(image_path)
        
        logger.info(f"Processing image: {image_path}")
        
        # Verify file was saved
        if not os.path.exists(image_path):
            return jsonify({
                "success": False,
                "message": "Failed to save image file"
            }), 500

        person_info, confidence = recognizer.recognize_face(image_path)
        
        # Clean up
        try:
            os.remove(image_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {e}")

        if person_info:
            return jsonify({
                "success": True,
                "id": person_info['id'],
                "role": person_info['role'],
                "confidence": float(confidence)
            })
        else:
            return jsonify({
                "success": False,
                "message": "Face not recognized or confidence too low"
            }), 400

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

if __name__ == "__main__":
    logger.info("\n=== Face Recognition Server Starting ===")
    logger.info("Server running on http://localhost:5000")
    logger.info("Waiting for recognition requests...")
    logger.info("=====================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False)