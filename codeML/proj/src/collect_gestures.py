# collect_gestures.py
import os, time, math, csv
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

LABEL_KEYS = {
    ord('1'): "OPEN",
    ord('2'): "FIST",
    ord('3'): "PINCH",
    ord('0'): None,     # pause
    ord('s'): "SNAP"    # special single-shot flag (we'll use current label)
}

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def _angle(a, b):
    return math.atan2(a[1]-b[1], a[0]-b[0])

def normalize_landmarks(lm_xy, handedness_label=None):
    """
    lm_xy: list of 21 (x,y) in image pixels or normalized [0,1].
    Steps:
      1) Mirror left to look like right (canonicalize).
      2) Translate so wrist (0) is at origin.
      3) Rotate so wrist->middle_MCP (0->9) is horizontal.
      4) Scale by that palm length to be ~1.
    Output: 42D vector [x0,y0, x1,y1, ...] in a canonical space.
    """
    pts = np.array(lm_xy, dtype=np.float32)  # shape (21,2)

    # 1) Mirror if Left so both hands look like Right
    if handedness_label and handedness_label.lower().startswith('l'):
        pts[:,0] = -pts[:,0]

    # 2) Translate so wrist at (0,0)
    wrist = pts[0].copy()
    pts -= wrist

    # 3) Rotate so vector (0->9) is horizontal
    axis = pts[9]  # middle MCP
    ang  = _angle(axis, np.array([0.0,0.0], dtype=np.float32))
    ca, sa = math.cos(-ang), math.sin(-ang)
    R = np.array([[ca, -sa],[sa, ca]], dtype=np.float32)
    pts = pts @ R.T

    # 4) Scale by palm length (wrist to middle MCP)
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale

    return pts.reshape(-1)  # 42 floats

def main():
    out_dir = Path("data"); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gestures.csv"
    if not out_csv.exists():
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(42)]
            w.writerow(header)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    current_label = None
    last_save = 0.0
    SAVE_EVERY_SEC = 0.12  # auto-save cadence when a label is active

    print("Controls: 1=OPEN, 2=FIST, 3=PINCH, 0=pause, S=single snapshot, ESC=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            img = cv2.flip(frame, 1)
            h, w = img.shape[:2]

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            handed_label = None
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                pts = [(p.x*w, p.y*h) for p in lm.landmark]
                mp_draw.draw_landmarks(
                    img, lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )
                if res.multi_handedness:
                    handed_label = res.multi_handedness[0].classification[0].label  # 'Left'/'Right'

                # Auto-save if a label is active and enough time passed
                now = time.time()
                if current_label and (now - last_save) >= SAVE_EVERY_SEC:
                    feat = normalize_landmarks(pts, handed_label)
                    with open(out_csv, "a", newline="") as f:
                        csv.writer(f).writerow([current_label] + feat.tolist())
                    last_save = now

            cv2.putText(img, f"Label: {current_label or 'NONE'}   (1=OPEN, 2=FIST, 3=PINCH, 0=pause, S=snap, ESC=quit)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Collect Gestures", img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            if k in LABEL_KEYS:
                if k == ord('s') and current_label and res.multi_hand_landmarks:
                    # one-shot save
                    pts = [(p.x*w, p.y*h) for p in res.multi_hand_landmarks[0].landmark]
                    handed_label = (res.multi_handedness[0].classification[0].label
                                    if res.multi_handedness else None)
                    feat = normalize_landmarks(pts, handed_label)
                    with open(out_csv, "a", newline="") as f:
                        csv.writer(f).writerow([current_label] + feat.tolist())
                else:
                    current_label = LABEL_KEYS[k]
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
