import cv2
import numpy as np
import os
import pickle
import logging
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class FaceTrainer:
    def __init__(self, firebase_cred_path, drive_cred_path):
        # Initialize Firebase
        try:
            firebase_cred = credentials.Certificate(firebase_cred_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(firebase_cred)
            self.db = firestore.client()
            logging.info("Firebase initialized successfully")
        except Exception as e:
            logging.error(f"Firebase initialization failed: {e}")
            raise

        # Initialize Google Drive
        try:
            SCOPES = [
                'https://www.googleapis.com/auth/drive.readonly',
                'https://www.googleapis.com/auth/drive.metadata.readonly'
            ]
            drive_credentials = service_account.Credentials.from_service_account_file(
                drive_cred_path, scopes=SCOPES)
            self.drive_service = build('drive', 'v3', credentials=drive_credentials)
            logging.info("Google Drive service initialized")
        except Exception as e:
            logging.error(f"Google Drive initialization failed: {e}")
            raise

        # OpenCV configurations
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Use scikit-learn for face recognition
        from sklearn.neighbors import KNeighborsClassifier
        self.face_recognizer = KNeighborsClassifier(n_neighbors=3)
        
        # Training data
        self.faces = []
        self.labels = []
        self.label_dict = {}
        self.user_ids = []  # Store user IDs for reference

    def list_all_images_recursively(self, folder_id):
        """Recursively list all image files in a folder and its subfolders"""
        image_files = []
        
        try:
            # First, list all items in the folder
            results = self.drive_service.files().list(
                q=f"'{folder_id}' in parents",
                spaces='drive',
                fields="files(id, name, mimeType)"
            ).execute()
            
            items = results.get('files', [])
            logging.info(f"Found {len(items)} items in the folder")
            
            for item in items:
                logging.debug(f"Processing item: {item['name']} (Type: {item['mimeType']})")
                
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    # If it's a folder, recursively list its contents
                    subfolder_images = self.list_all_images_recursively(item['id'])
                    image_files.extend(subfolder_images)
                elif item['mimeType'].startswith('image/'):
                    # If it's an image, add to the list
                    image_files.append({
                        'id': item['id'], 
                        'name': item['name']
                    })
            
            logging.info(f"Total images found: {len(image_files)}")
            return image_files
        
        except Exception as e:
            logging.error(f"Error listing images recursively: {e}")
            return []

    def verify_folder_exists(self, folder_id):
        """Verify if the folder exists and you have access"""
        try:
            folder = self.drive_service.files().get(fileId=folder_id).execute()
            logging.info(f"Folder found: {folder.get('name', 'Unnamed Folder')}")
            return True
        except Exception as e:
            logging.error(f"Folder verification failed: {e}")
            return False

    def download_image_from_drive(self, file_id):
        """Download image from Google Drive"""
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file.seek(0)
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logging.error(f"Image download error: {e}")
            return None

    def get_user_data(self):
        """Fetch user data from Firebase"""
        try:
            users_ref = self.db.collection('users')
            users = users_ref.stream()
            
            for idx, user in enumerate(users):
                user_data = user.to_dict()
                user_id = user.id
                self.user_ids.append(user_id)
                self.label_dict[user_id] = {
                    "index": idx,
                    "name": user_data.get('name', 'Unknown'),
                    "role": user_data.get('role', 'unknown')
                }
            
            logging.info(f"Fetched {len(self.label_dict)} users from Firebase")
            return True
        except Exception as e:
            logging.error(f"Error fetching user data: {e}")
            return False

    def match_user_id(self, filename):
        """
        Try to match the filename with a user ID
        Supports multiple matching strategies
        """
        # Strategy 1: Exact match with user IDs
        for user_id in self.user_ids:
            if user_id in filename:
                return user_id
        
        # Strategy 2: Regex matching for common ID patterns
        patterns = [
            r'(21\d{6})',  # Match student IDs like 21551A0533
            r'(CSE\d+)',    # Match teacher/staff IDs like CSE1456
            r'(MCA\d+)',    # Match MCA student IDs
            r'(DEG\d+)'     # Match other degree student IDs
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                potential_id = match.group(1)
                if potential_id in self.user_ids:
                    return potential_id
        
        return None

    def preprocess_image(self, img):
        """Detect and extract faces from an image"""
        try:
            if img is None:
                return []

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhance image quality
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces_detected = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Extract and resize faces
            face_regions = []
            for (x, y, w, h) in faces_detected:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (200, 200))
                face_regions.append(face_resized)
            
            return face_regions
        except Exception as e:
            logging.error(f"Image preprocessing error: {e}")
            return []

    def train_model(self, drive_folder_id):
        """Train face recognition model"""
        # Verify folder exists
        if not self.verify_folder_exists(drive_folder_id):
            logging.error("Folder does not exist or is inaccessible")
            return False

        # Get user data from Firebase
        if not self.get_user_data():
            logging.error("Failed to retrieve user data")
            return False

        # List images in Drive folder
        drive_images = self.list_all_images_recursively(drive_folder_id)
        
        logging.info(f"Total images found: {len(drive_images)}")
        
        # Process dataset
        for img_info in drive_images:
            filename = img_info['name']
            
            # Try to match user ID
            user_id = self.match_user_id(filename)
            
            if user_id:
                # Download image
                img = self.download_image_from_drive(img_info['id'])
                
                # Detect and process faces
                face_regions = self.preprocess_image(img)
                
                for face in face_regions:
                    self.faces.append(face.flatten())  # Flatten for scikit-learn
                    self.labels.append(self.label_dict[user_id]['index'])
                
                logging.debug(f"Processed image for user {user_id}: {filename}")
            else:
                logging.warning(f"Could not match user ID for image: {filename}")

        # Train model
        if len(self.faces) >= 5:
            try:
                # Train the model
                self.face_recognizer.fit(self.faces, self.labels)
                
                # Create model directory
                os.makedirs("model", exist_ok=True)
                
                # Save model and label dictionary
                model_path = "model/face_recognition_model.pkl"
                label_path = "model/label_dict.pkl"
                
                # Save the model
                import pickle
                with open(model_path, "wb") as f:
                    pickle.dump(self.face_recognizer, f)
                
                # Save label dictionary
                with open(label_path, "wb") as f:
                    pickle.dump(self.label_dict, f)
                
                logging.info(f"Training completed successfully!")
                logging.info(f"Total faces processed: {len(self.faces)}")
                logging.info(f"Model saved to: {model_path}")
                return True
                
            except Exception as e:
                logging.error(f"Model training error: {e}")
                return False
        else:
            logging.error(f"Insufficient faces for training. Found {len(self.faces)}, minimum required: 5")
            return False

def main():
    # Paths to credentials - replace with your actual paths
    FIREBASE_CRED_PATH = "C:/CertifySecure/certify-36ea0-firebase-adminsdk-uekjq-37e7e9448b.json"
    DRIVE_CRED_PATH = "C:/face-recognition-app/face-recognition-app-450504-fbfd29689a93.json"
    
    # Folder ID for the specific user's folder
    DRIVE_FOLDER_ID = "1D1-3yw-JMaEfYv_ipyWn6HmkmZRGMyMb"

    try:
        trainer = FaceTrainer(FIREBASE_CRED_PATH, DRIVE_CRED_PATH)
        success = trainer.train_model(DRIVE_FOLDER_ID)
        
        if success:
            logging.info("Face recognition model training completed successfully")
        else:
            logging.error("Face recognition model training failed")
    
    except Exception as e:
        logging.error(f"Training process failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()