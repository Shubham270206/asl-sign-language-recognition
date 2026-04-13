import subprocess
import sys
from pathlib import Path

def check_requirements():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        print(f"Missing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

def check_model():
    if not Path("models/asl_model.pkl").exists():
        print("No trained model found.")
        print("Running training pipeline first...\n")

        if not Path("data/landmarks/landmarks.csv").exists():
            print("No landmark CSV found.")
            print("Please run: python src/extract_landmarks.py")
            sys.exit(1)

        subprocess.run([sys.executable, "src/retrain_model.py"], check=True)

if __name__ == "__main__":
    print("=" * 50)
    print("   ASL Sign Language Recognition")
    print("=" * 50)

    check_requirements()
    check_model()

    print("\nStarting webcam inference...")
    print("Controls: SPACE = clear sentence | Q = quit\n")
    subprocess.run([sys.executable, "src/inference.py"])