🧠 ASL Sign Language Recognition (Real-Time)

A real-time American Sign Language (ASL) recognition system using a webcam.
This project detects hand landmarks with MediaPipe and classifies them into 29 ASL signs (A–Z + space, delete, nothing) using a trained Random Forest model.

🚀 Overview

This system enables seamless gesture-to-text conversion by interpreting hand signs captured through a webcam in real time.

Pipeline:

Webcam → Hand Detection → Landmark Extraction → Feature Normalization → ML Model → Character Output
🎯 Features
🔴 Real-time ASL recognition (30 FPS)
✋ Detects 21 hand landmarks (63 features per frame)
🌲 Random Forest model (150 trees)
🧠 Trained on 95,000+ samples
🔄 Prediction smoothing to reduce flickering
⏱️ Dwell-time based character input (~0.7 sec hold)
⌨️ Interactive controls for text building
🖥️ Demo

Show an ASL sign in front of your webcam → hold steady → predicted letter appears on screen.

⚙️ How It Works
Hand Detection
MediaPipe extracts 21 keypoints (x, y, z coordinates)
Feature Engineering
Normalize landmarks relative to wrist
Removes dependency on position and scale
Model Prediction
Input: 63 features
Model: Random Forest
Output: One of 29 classes
Post-Processing
Frame smoothing (last 5 predictions)
Dwell timer to confirm stable gestures
📁 Project Structure
asl-sign-language-recognition/
│
├── data/
│   ├── raw/               # Original dataset (not included)
│   └── landmarks/         # Extracted features (CSV)
│
├── models/                # Trained model files (.pkl)
│
├── src/
│   ├── extract_landmarks.py   # Extract MediaPipe features
│   ├── train_model.py         # Train on dataset
│   ├── collect_data.py        # Capture custom gestures
│   ├── retrain_model.py       # Train on combined dataset
│   └── inference.py           # Real-time prediction
│
├── run.py                 # Main execution script
├── requirements.txt
└── README.md
⚡ Installation & Setup
1. Clone Repository
git clone https://github.com/Shubham270206/asl-sign-language-recognition.git
cd asl-sign-language-recognition
2. Create Virtual Environment
python -m venv venv

# Activate
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
3. Install Dependencies
pip install -r requirements.txt
📦 Dataset

Download the dataset from Kaggle:

👉 https://www.kaggle.com/datasets/grassknoted/asl-alphabet

~87,000 images
29 classes
Place in:
data/raw/asl_alphabet_train/

▶️ Run the Project
Step 1: Extract Features
python src/extract_landmarks.py
Step 2: Train Model
python src/retrain_model.py
Step 3: Start Real-Time Inference
python run.py


🎮 Controls
Key	Action
SPACE	Clear sentence
Q	Quit application


📊 Results
Metric	Value
Dataset Size	95,272 samples
Classes	29
Model	Random Forest
Accuracy	99.25%
Speed	~30 FPS


🛠️ Tech Stack
MediaPipe → Hand tracking
OpenCV → Video processing
Scikit-learn → ML model
NumPy / Pandas → Data handling


🧭 Future Improvements
 Streamlit/Web UI deployment
 Word-level prediction (NLP integration)
 Two-hand gesture recognition
 Mobile app deployment
 Deep learning model (CNN/LSTM)
👨‍💻 Author

Shubham Sinha
🔗 GitHub: https://github.com/Shubham270206

📜 License

This project is licensed under the MIT License.