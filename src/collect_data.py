import cv2
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path

# ── Settings ───────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 20
OUTPUT_CSV        = Path("data/landmarks/my_landmarks.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

CLASSES = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','nothing','space'
]

COLORS = {
    "green":  (0, 255, 150),
    "red":    (0, 0, 255),
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "yellow": (0, 220, 255),
    "dim":    (100, 100, 100),
}

# ── MediaPipe setup ────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Helpers ────────────────────────────────────────────────────────
def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    origin = coords[0]
    coords -= origin
    scale  = np.max(np.abs(coords))
    if scale > 0:
        coords /= scale
    return coords.flatten().tolist()

def draw_progress_bar(frame, x, y, w, h, value, maximum, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS["dim"], -1)
    filled = int(w * value / maximum)
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.putText(frame, f"{value}/{maximum}", (x + w + 10, y + h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["white"], 1)

# ── Check for existing progress ────────────────────────────────────
completed = set()
if OUTPUT_CSV.exists():
    import pandas as pd
    existing = pd.read_csv(OUTPUT_CSV)
    for label, count in existing["label"].value_counts().items():
        if count >= SAMPLES_PER_CLASS:
            completed.add(label)
    print(f"Resuming — already completed: {sorted(completed)}")

# ── CSV setup ──────────────────────────────────────────────────────
header = ["label"] + [f"{a}{i}" for i in range(21) for a in ["x","y","z"]]
file_exists = OUTPUT_CSV.exists()
csv_file = open(OUTPUT_CSV, "a", newline="")
writer   = csv.writer(csv_file)
if not file_exists:
    writer.writerow(header)

# ── Main collection loop ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for class_idx, label in enumerate(CLASSES):
    if label in completed:
        print(f"Skipping '{label}' — already collected")
        continue

    count        = 0
    collecting   = False
    countdown    = 0

    print(f"\n>>> Get ready for: '{label}' ({class_idx+1}/{len(CLASSES)})")

    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_detected = False

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            hand_detected = True

            if collecting:
                coords = normalize_landmarks(hand_lm.landmark)
                writer.writerow([label] + coords)
                csv_file.flush()
                count += 1

        # ── Overlay ───────────────────────────────────────────────
        # Dark top panel
        cv2.rectangle(frame, (0, 0), (w, 130), COLORS["black"], -1)

        # Class name
        cv2.putText(frame, f"Sign:  {label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    COLORS["yellow"], 2)

        # Progress bar
        draw_progress_bar(frame, 20, 70, 500, 18,
                          count, SAMPLES_PER_CLASS,
                          COLORS["green"] if collecting else COLORS["dim"])

        # Status
        if not hand_detected:
            status_text  = "No hand detected — show your hand"
            status_color = COLORS["red"]
        elif not collecting:
            status_text  = "Press SPACE to start recording"
            status_color = COLORS["yellow"]
        else:
            status_text  = f"Recording... {count}/{SAMPLES_PER_CLASS}"
            status_color = COLORS["green"]

        cv2.putText(frame, status_text, (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 1)

        # Overall progress bottom right
        cv2.putText(frame, f"Class {class_idx+1}/{len(CLASSES)}",
                    (w - 200, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["dim"], 1)

        # Next class hint
        if class_idx + 1 < len(CLASSES):
            cv2.putText(frame, f"Next: {CLASSES[class_idx+1]}",
                        (w - 200, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["dim"], 1)

        # Controls
        cv2.putText(frame, "SPACE: record   S: skip   Q: quit",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["dim"], 1)

        cv2.imshow("ASL Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit — progress saved.")
            csv_file.close()
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()
        elif key == ord(" "):
            if hand_detected:
                collecting = not collecting
                if not collecting:
                    print(f"  Paused at {count} samples")
            else:
                print("  No hand detected — can't start recording")
        elif key == ord("s"):
            print(f"  Skipped '{label}'")
            break

    if count >= SAMPLES_PER_CLASS:
        print(f"  Done '{label}' — {count} samples collected")
        collecting = False

csv_file.close()
cap.release()
cv2.destroyAllWindows()
hands.close()
print("\nAll done! Data saved to:", OUTPUT_CSV)
print("Now run: python src/retrain_model.py")