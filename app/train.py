import cv2
import numpy as np
import os
import pickle
import logging
import firebase_admin
from firebase_admin import credentials, firestore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class FaceTrainer:
    def __init__(self):
        # Get Firebase credentials path from environment variable
         firebase_cred_path = os.getenv('FIREBASE_CRED_PATH', '/etc/secrets/firebase-credentials.json')
        
        if not firebase_cred_path:
            logging.error("FIREBASE_CRED_PATH environment variable not set")
            exit(1)

        # Initialize Firebase
        try:
            cred = credentials.Certificate(firebase_cred_path)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logging.info("Firebase initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Firebase: {e}")
            exit(1)

        # Initialize OpenCV face detector and recognizer
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faces = []
        self.labels = []
        self.label_dict = {}

    def get_user_data(self):
        """Fetch user data from Firebase"""
        try:
            users_ref = self.db.collection('users')
            users = users_ref.stream()
            
            for idx, user in enumerate(users):
                user_data = user.to_dict()
                if 'role' in user_data:
                    self.label_dict[idx] = {
                        "id": user.id,
                        "role": user_data['role']
                    }
            
            logging.info(f"Fetched {len(self.label_dict)} users from Firebase")
            return True
        except Exception as e:
            logging.error(f"Error fetching user data: {e}")
            return False

    def preprocess_image(self, img_path):
        """Detect and extract faces from an image"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.warning(f"Could not read image: {img_path}")
                return []

            # Enhance image quality
            img = cv2.equalizeHist(img)
            
            faces_detected = self.face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return [cv2.resize(img[y:y+h, x:x+w], (200, 200)) for (x, y, w, h) in faces_detected]
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
            return []

    def train_model(self):
        """Train the face recognition model"""
        # Get user data from Firebase
        if not self.get_user_data():
            return False

        # Process dataset for each user
        for label_idx, user_info in self.label_dict.items():
            user_dataset_path = f"dataset/{user_info['id']}"
            
            if not os.path.exists(user_dataset_path):
                logging.warning(f"No dataset found for user {user_info['id']}")
                continue
            
            # Process each image in user's dataset
            for img_file in os.listdir(user_dataset_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(user_dataset_path, img_file)
                face_regions = self.preprocess_image(img_path)
                
                if face_regions:
                    for face in face_regions:
                        self.faces.append(face)
                        self.labels.append(label_idx)
                else:
                    logging.warning(f"No faces detected in {img_file}")

        # Train and save model if faces were detected
        if len(self.faces) >= 5:  # Minimum faces required for training
            try:
                self.face_recognizer.train(self.faces, np.array(self.labels))
                
                # Create model directory if it doesn't exist
                os.makedirs("model", exist_ok=True)
                
                # Save model and label dictionary
                model_path = "model/face_recognition_model.yml"
                label_path = "model/label_dict.pkl"
                
                self.face_recognizer.save(model_path)
                with open(label_path, "wb") as f:
                    pickle.dump(self.label_dict, f)
                
                logging.info(f"Training completed successfully!")
                logging.info(f"Total faces processed: {len(self.faces)}")
                logging.info(f"Model saved to: {model_path}")
                logging.info(f"Label dictionary saved to: {label_path}")
                return True
                
            except Exception as e:
                logging.error(f"Error during model training: {e}")
                return False
        else:
            logging.error(f"Insufficient faces for training. Found {len(self.faces)}, minimum required: 5")
            return False

def main():
    trainer = FaceTrainer()
    success = trainer.train_model()
    if success:
        logging.info("Training process completed successfully")
    else:
        logging.error("Training process failed")

if __name__ == "__main__":
    main()