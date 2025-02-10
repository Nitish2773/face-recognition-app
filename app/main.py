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

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

class FaceRecognizer:
    def __init__(self):
        try:
            # Check if running on Render
            if os.getenv('RENDER'):
                # Use Render's secret file path
                cred_path = '/etc/secrets/firebase-credentials.json'
            else:
                # Local development path
                cred_path = './secrets/firebase-credentials.json'

            print(f"Looking for Firebase credentials at: {cred_path}")
            
            if not os.path.exists(cred_path):
                raise FileNotFoundError(f"Firebase credentials not found at {cred_path}")

            cred = credentials.Certificate(cred_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("Firebase initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Firebase: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            exit(1)
        # Get model paths from environment variables or use defaults
        self.MODEL_PATH = os.getenv('MODEL_PATH', "model/face_recognition_model.yml")
        self.LABEL_DICT_PATH = os.getenv('LABEL_DICT_PATH', "model/label_dict.pkl")
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.load_recognizer()


    def load_recognizer(self):
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.read(self.MODEL_PATH)
            with open(self.LABEL_DICT_PATH, "rb") as f:
                self.label_dict = pickle.load(f)
            print("Model and label dictionary loaded successfully")
        except Exception as e:
            print(f"Error loading model or label dictionary: {e}")
            exit(1)

    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read image: {image_path}")
                return None

            img = cv2.equalizeHist(img)
            faces_detected = self.face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces_detected) == 0:
                print("No faces detected in image")
                return None

            x, y, w, h = faces_detected[0]
            face = cv2.resize(img[y:y+h, x:x+w], (200, 200))
            return face

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def recognize_face(self, image_path):
        face = self.preprocess_image(image_path)
        if face is None:
            return None, None

        try:
            label, confidence = self.face_recognizer.predict(face)
            person_info = self.label_dict.get(label, {"id": "Unknown", "role": "Unknown"})
            
            print(f"\n=== Recognition Attempt ===")
            print(f"ID: {person_info['id']}")
            print(f"Confidence: {confidence}")
            
            user_ref = self.db.collection('users').document(person_info['id'])
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                person_info['role'] = user_data.get('role', 'Unknown')
                
                if confidence > 200:
                    print("Recognition Failed: Confidence too low")
                    return None, None
                    
                print(f"Recognition Successful!")
                print(f"Role: {person_info['role']}")
                print("========================\n")
                return person_info, confidence
            else:
                print(f"User {person_info['id']} not found in Firebase")
                return None, None

        except Exception as e:
            print(f"Error during face recognition: {e}")
            return None, None

recognizer = FaceRecognizer()

@app.route('/recognize', methods=['POST'])
def api_recognize():
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "message": "No image file provided"
        }), 400

    try:
        image_file = request.files['image']
        os.makedirs('temp', exist_ok=True)
        image_path = os.path.join('temp', image_file.filename)
        image_file.save(image_path)

        person_info, confidence = recognizer.recognize_face(image_path)
        os.remove(image_path)

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
        print(f"Error processing request: {e}")
        return jsonify({
            "success": False,
            "message": "Internal server error"
        }), 500

if __name__ == "__main__":
    print("\n=== Face Recognition Server Starting ===")
    print("Server running on http://localhost:5000")
    print("Waiting for recognition requests...")
    print("=====================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False)