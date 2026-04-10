import cv2
import pickle
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque, Counter

# ── Load model + encoder ───────────────────────────────────────────
MODEL_PATH   = Path("models/asl_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# ── MediaPipe setup ────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,      # video mode — faster
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Settings ───────────────────────────────────────────────────────
SMOOTHING_WINDOW  = 5      # frames to smooth prediction over
CONFIDENCE_THRESH = 0.70   # below this shows "?"
DWELL_FRAMES      = 20     # stable frames before appending letter
COLORS = {
    "box":    (0, 255, 150),
    "text":   (0, 255, 150),
    "dim":    (100, 100, 100),
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "red":    (0, 0, 255),
}

# ── Helpers ────────────────────────────────────────────────────────
def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    origin = coords[0]
    coords -= origin
    scale = np.max(np.abs(coords))
    if scale > 0:
        coords /= scale
    return coords.flatten().reshape(1, -1)

def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=10, thickness=-1):
    """Draw a filled rounded rectangle."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

# ── State ──────────────────────────────────────────────────────────
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
sentence          = []       # accumulated letters
current_word      = ""
dwell_counter     = 0
last_stable_letter = None

# ── Webcam loop ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("ASL Recognition running — press Q to quit, SPACE to clear sentence")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)   # mirror so it feels natural
    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    letter      = None
    confidence  = 0.0

    if result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # Bounding box
        xs = [lm.x * w for lm in hand_lm.landmark]
        ys = [lm.y * h for lm in hand_lm.landmark]
        x1, y1 = max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20)
        x2, y2 = min(w, int(max(xs)) + 20), min(h, int(max(ys)) + 20)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["box"], 2)

        # Predict
        features = normalize_landmarks(hand_lm.landmark)
        proba    = model.predict_proba(features)[0]
        top_idx  = np.argmax(proba)
        confidence = proba[top_idx]

        if confidence >= CONFIDENCE_THRESH:
            letter = le.classes_[top_idx]
            prediction_buffer.append(letter)
        else:
            prediction_buffer.append(None)

    # Smoothed prediction
    valid = [p for p in prediction_buffer if p is not None]
    stable_letter = Counter(valid).most_common(1)[0][0] if valid else None

    # Dwell logic — append letter only after DWELL_FRAMES of same letter
    if stable_letter and stable_letter not in ("nothing",):
        if stable_letter == last_stable_letter:
            dwell_counter += 1
        else:
            dwell_counter     = 0
            last_stable_letter = stable_letter

        if dwell_counter == DWELL_FRAMES:
            if stable_letter == "space":
                sentence.append(" ")
            elif stable_letter == "del":
                if sentence:
                    sentence.pop()
            else:
                sentence.append(stable_letter)
            dwell_counter = 0
    else:
        dwell_counter = 0

    # ── HUD ───────────────────────────────────────────────────────
    # Top bar background
    draw_rounded_rect(frame, 10, 10, 420, 110, COLORS["black"], radius=12)

    # Current letter
    disp_letter = stable_letter if stable_letter else "?"
    cv2.putText(frame, disp_letter, (28, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, COLORS["text"], 3)

    # Confidence bar
    bar_x, bar_y = 130, 35
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 260, bar_y + 16),
                  COLORS["dim"], -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(260 * confidence), bar_y + 16),
                  COLORS["text"], -1)
    cv2.putText(frame, f"{confidence * 100:.0f}%", (bar_x + 268, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)

    # Dwell progress dots
    for i in range(DWELL_FRAMES):
        cx = 130 + i * (260 // DWELL_FRAMES)
        color = COLORS["text"] if i < dwell_counter else COLORS["dim"]
        cv2.circle(frame, (cx, bar_y + 36), 3, color, -1)

    # Sentence bar at bottom
    sentence_str = "".join(sentence)
    draw_rounded_rect(frame, 10, h - 60, w - 10, h - 10,
                      COLORS["black"], radius=10)
    cv2.putText(frame, sentence_str[-60:], (24, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS["white"], 2)

    # Controls hint
    cv2.putText(frame, "Q: quit   SPACE: clear", (w - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["dim"], 1)

    cv2.imshow("ASL Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        sentence.clear()
        dwell_counter = 0

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Session ended.")