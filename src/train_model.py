import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────
CSV_PATH   = Path("data/landmarks/landmarks.csv")
MODEL_PATH = Path("models/asl_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
print("Loading landmarks CSV...")
df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Classes: {sorted(df['label'].unique())}")

X = df.drop("label", axis=1).values
y = df["label"].values

# ── Encode labels ──────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEncoded {len(le.classes_)} classes")

# ── Train / test split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded   # equal representation of all classes
)
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# ── Train Random Forest ────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1           # use all CPU cores
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────
print("\nEvaluating...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nPer-class report:")
print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=le.classes_[np.unique(y_test)]))

# ── Save model + encoder ───────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Label encoder saved to: {ENCODER_PATH}")