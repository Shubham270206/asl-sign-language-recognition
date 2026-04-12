import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────
KAGGLE_CSV   = Path("data/landmarks/landmarks.csv")
MY_CSV       = Path("data/landmarks/my_landmarks.csv")
MODEL_PATH   = Path("models/asl_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")

# ── Load and combine data ──────────────────────────────────────────
print("Loading Kaggle landmarks...")
df_kaggle = pd.read_csv(KAGGLE_CSV)
print(f"Kaggle samples: {len(df_kaggle)}")

print("Loading personal landmarks...")
df_mine = pd.read_csv(MY_CSV)
print(f"Personal samples: {len(df_mine)}")

# Upsample personal data — repeat it 50x so model pays attention to it
df_mine_upsampled = pd.concat([df_mine] * 50, ignore_index=True)
print(f"Personal samples after 50x upsample: {len(df_mine_upsampled)}")

df = pd.concat([df_kaggle, df_mine_upsampled], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print(f"\nCombined dataset: {len(df)} samples")

X = df.drop("label", axis=1).values
y = df["label"].values

# ── Encode labels ──────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── Train / test split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train ──────────────────────────────────────────────────────────
print("\nTraining Random Forest on combined data...")
model = RandomForestClassifier(
    n_estimators=150,   # slightly more trees than before
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nPer-class report:")
print(classification_report(
    y_test, y_pred,
    labels=np.unique(y_test),
    target_names=le.classes_[np.unique(y_test)]
))

# ── Save ───────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)

print(f"Model saved to: {MODEL_PATH}")
print(f"Encoder saved to: {ENCODER_PATH}")