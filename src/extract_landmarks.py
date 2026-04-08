import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# ── MediaPipe setup ────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,   # we're processing images, not video
    max_num_hands=1,
    min_detection_confidence=0.3
)

# ── Paths ──────────────────────────────────────────────────────────
TRAIN_DIR = Path("data/raw/asl_alphabet_train/asl_alphabet_train")
OUTPUT_CSV = Path("data/landmarks/landmarks.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── CSV header: label + 21 landmarks × (x, y, z) ──────────────────
header = ["label"]
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]

# ── Extract ────────────────────────────────────────────────────────
def normalize_landmarks(landmarks):
    """Translate so wrist (landmark 0) is origin, then scale by hand size."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    origin = coords[0]               # wrist
    coords -= origin                 # center on wrist
    scale = np.max(np.abs(coords))   # normalize scale
    if scale > 0:
        coords /= scale
    return coords.flatten().tolist()

skipped = 0
written = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    classes = sorted(os.listdir(TRAIN_DIR))
    print(f"Found {len(classes)} classes: {classes}\n")

    for label in classes:
        class_dir = TRAIN_DIR / label
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"Processing '{label}' — {len(images)} images...")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            # MediaPipe expects RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if not result.multi_hand_landmarks:
                skipped += 1
                continue

            coords = normalize_landmarks(result.multi_hand_landmarks[0].landmark)
            writer.writerow([label] + coords)
            written += 1

hands.close()
print(f"\nDone! Rows written: {written} | Images skipped: {skipped}")
print(f"CSV saved to: {OUTPUT_CSV}")