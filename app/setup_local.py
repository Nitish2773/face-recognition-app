import os
import shutil

def setup_local_environment():
    # Create necessary directories
    directories = ['secrets', 'model', 'temp']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created {directory} directory")

    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("PORT=5000\n")
            f.write("RENDER=false\n")
        print("Created .env file")

    # Check for Firebase credentials
    if not os.path.exists('secrets/firebase-credentials.json'):
        print("\nIMPORTANT: Please add your Firebase credentials:")
        print("1. Copy your Firebase service account JSON file")
        print("2. Place it at: secrets/firebase-credentials.json")

if __name__ == "__main__":
    setup_local_environment()