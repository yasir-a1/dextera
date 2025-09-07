import argparse
import os
import time
from typing import Tuple

import cv2
import numpy as np
import mediapipe as mp

# Short aliases
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def put_text(img, text, org, scale=0.9, color=(0, 255, 0), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_handedness(img, lm, handedness, color=(0, 255, 0)):
    """Draw Left/Right label near the wrist."""
    h, w = img.shape[:2]
    wrist = lm.landmark[0]
    x, y = int(wrist.x * w), int(wrist.y * h)
    label = handedness.classification[0].label  # 'Left' or 'Right'
    score = handedness.classification[0].score
    put_text(img, f"{label} ({score:.2f})", (x + 10, y - 10), 0.7, color, 2)

def main():
    ap = argparse.ArgumentParser(description="Real-time MediaPipe Hands landmark viewer")
    ap.add_argument("--camera",    type=int, default=0, help="Camera index (0 is default)")
    ap.add_argument("--width",     type=int, default=1280, help="Capture width")
    ap.add_argument("--height",    type=int, default=720,  help="Capture height")
    ap.add_argument("--max-hands", type=int, default=1,    help="Max number of hands to track")
    ap.add_argument("--complexity",type=int, default=0,    choices=[0,1], help="Model complexity: 0=fast, 1=accurate")
    ap.add_argument("--det-conf",  type=float, default=0.6, help="Min detection confidence")
    ap.add_argument("--track-conf",type=float, default=0.6, help="Min tracking confidence")
    args = ap.parse_args()

    # Prefer DirectShow on Windows for fewer webcam issues, else default
    api_pref = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, api_pref)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    # Mediapipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=args.complexity,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf
    )

    fps = 0.0
    t_prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror for nicer UX
            img = cv2.flip(frame, 1)

            # BGR -> RGB for MediaPipe
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # Draw results
            if res.multi_hand_landmarks:
                # multi_hand_landmarks and multi_handedness are aligned lists
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness or []):
                    mp_draw.draw_landmarks(
                        img,
                        lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    draw_handedness(img, lm, handed, (0, 255, 0))

            # FPS
            t = time.time()
            dt = max(t - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # EMA smoothing
            t_prev = t
            put_text(img, f"FPS: {fps:.1f}   (ESC to quit)", (10, 30), 0.8, (255, 255, 255), 2)

            cv2.imshow("MediaPipe Hands — Landmarks", img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()