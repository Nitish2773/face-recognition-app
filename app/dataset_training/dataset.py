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
        """Create dataset for a specific user"""
        # Create directory for user
        dataset_path = f"dataset/{user_id}"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        count = 0
        print(f"Starting capture for user {user_id}")
        print("Press 'q' to quit capturing")

        while count < num_images:
            ret, frame = self.cap.read()
            if not ret:
                print("Error capturing frame")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face = gray[y:y+h, x:x+w]
                
                # Resize to standard size
                face_resized = cv2.resize(face, (200, 200))
                
                # Apply histogram equalization
                face_enhanced = cv2.equalizeHist(face_resized)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{dataset_path}/{timestamp}.jpg"
                
                # Save the image
                cv2.imwrite(filename, face_enhanced)
                count += 1
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captures: {count}/{num_images}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Capturing Face Data", frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Dataset creation completed. {count} images captured.")
        
    def augment_dataset(self, user_id):
        """Augment existing dataset with variations"""
        source_path = f"dataset/{user_id}"
        augmented_path = f"dataset/{user_id}/augmented"
        
        if not os.path.exists(augmented_path):
            os.makedirs(augmented_path)
            
        print(f"Augmenting dataset for user {user_id}")
        
        for img_name in os.listdir(source_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(source_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # Original image
            cv2.imwrite(f"{augmented_path}/orig_{img_name}", img)
            
            # Flipped image
            flipped = cv2.flip(img, 1)
            cv2.imwrite(f"{augmented_path}/flip_{img_name}", flipped)
            
            # Rotated images
            for angle in [-15, 15]:
                h, w = img.shape
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                rotated = cv2.warpAffine(img, M, (w, h))
                cv2.imwrite(f"{augmented_path}/rot{angle}_{img_name}", rotated)
            
            # Brightness variations
            for alpha in [0.8, 1.2]:
                adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                cv2.imwrite(f"{augmented_path}/bright{alpha}_{img_name}", adjusted)
                
        print("Dataset augmentation completed")

def main():
    creator = DatasetCreator()
    
    while True:
        print("\nDataset Creation Menu:")
        print("1. Capture new dataset")
        print("2. Augment existing dataset")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            user_id = input("Enter user ID: ")
            num_images = int(input("Enter number of images to capture (default 500): ") or "500")
            creator.create_user_dataset(user_id, num_images)
            
        elif choice == '2':
            user_id = input("Enter user ID to augment: ")
            creator.augment_dataset(user_id)
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()