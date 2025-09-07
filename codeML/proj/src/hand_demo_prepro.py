# hand_landmarks_canonical_overlay.py
import argparse
import os
import time
import math
from typing import Tuple, List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Short aliases
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ---------- UI helpers ----------
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

# ---------- Geometry / preprocessing ----------
def _angle(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    """Angle (radians) of vector a->b relative to +x axis."""
    v = b_xy - a_xy
    return math.atan2(v[1], v[0])

def normalize_landmarks(lm_xy: np.ndarray, handed_label: str | None) -> np.ndarray:
    """
    Canonicalize 21 (x,y) landmarks to be invariant to:
      - handedness (mirror Left -> Right)
      - translation (center at wrist)
      - rotation (align wrist->middle MCP horizontally)
      - scale (divide by palm length)
    Args:
      lm_xy: shape (21, 2) in pixel coords (or any consistent units)
      handed_label: 'Left' / 'Right' / None
    Returns:
      flat 42-D vector (x0,y0, x1,y1, ..., x20,y20) in canonical space
    """
    pts = lm_xy.astype(np.float32).copy()  # (21,2)

    # 1) Mirror if Left so both hands look like Right
    if handed_label and handed_label.lower().startswith('l'):
        pts[:, 0] = -pts[:, 0]

    # 2) Translate so wrist at origin
    wrist = pts[0].copy()
    pts -= wrist

    # 3) Rotate so wrist->middle MCP (landmark 9) is horizontal
    #    (0 -> 9) defines palm axis
    palm_axis = pts[9]
    ang = math.atan2(palm_axis[1], palm_axis[0])  # radians
    ca, sa = math.cos(-ang), math.sin(-ang)
    R = np.array([[ca, -sa],
                  [sa,  ca]], dtype=np.float32)
    pts = pts @ R.T

    # 4) Scale by palm length (||wrist->middle MCP||)
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale

    return pts.reshape(-1)  # 42 numbers

# ---------- Drawing canonical landmarks as an inset ----------
def draw_canonical_inset(
    img: np.ndarray,
    canon_vec: np.ndarray,
    inset_anchor: Tuple[int, int],
    inset_size: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 0, 255),
    box_color: Tuple[int, int, int] = (255, 0, 255),
    label: str = "Canonical (preprocessed)"
) -> None:
    """
    Render the canonicalized landmarks as a small inset in the frame using mp_draw.draw_landmarks.
    We assume canonical coordinates roughly live within [-2, 2] both in x and y.
    """
    ih, iw = img.shape[:2]
    x0, y0 = inset_anchor
    w, h   = inset_size

    # Draw inset border & label
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), box_color, 1)
    put_text(img, label, (x0 + 6, y0 + 20), 0.6, box_color, 1)

    # Convert 42-D vector back to (21,2) canonical coords
    pts = canon_vec.reshape(21, 2)

    # Map canonical coords (assumed in [-R, R]) into the inset rect
    R = 2.0  # canonical range half-extent; 99% of poses fit within [-2, 2]
    xs = (pts[:, 0] - (-R)) / (2 * R)  # -> [0,1]
    ys = (pts[:, 1] - (-R)) / (2 * R)  # -> [0,1]

    # Clamp to [0,1] to avoid drawing outside inset
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)

    # Convert to image-normalized coordinates (0..1 across full image)
    nx = (x0 + xs * w) / float(iw)
    ny = (y0 + ys * h) / float(ih)

    # Build a MediaPipe NormalizedLandmarkList
    nlms = []
    for xi, yi in zip(nx, ny):
        nlms.append(landmark_pb2.NormalizedLandmark(x=float(xi), y=float(yi), z=0.0))
    nlm_list = landmark_pb2.NormalizedLandmarkList(landmark=nlms)

    # Custom drawing specs (magenta)
    lspec = mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2)
    cspec = mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2)

    # Draw the canonical skeleton inside the inset
    mp_draw.draw_landmarks(
        image=img,
        landmark_list=nlm_list,
        connections=mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=lspec,
        connection_drawing_spec=cspec,
    )

def main():
    ap = argparse.ArgumentParser(description="Realtime MediaPipe Hands with canonicalized 42-D overlay")
    ap.add_argument("--camera",    type=int, default=0, help="Camera index (0 is default)")
    ap.add_argument("--width",     type=int, default=1280, help="Capture width")
    ap.add_argument("--height",    type=int, default=720,  help="Capture height")
    ap.add_argument("--max-hands", type=int, default=1,    help="Max number of hands to track")
    ap.add_argument("--complexity",type=int, default=0,    choices=[0,1], help="Model complexity: 0=fast, 1=accurate")
    ap.add_argument("--det-conf",  type=float, default=0.6, help="Min detection confidence")
    ap.add_argument("--track-conf",type=float, default=0.6, help="Min tracking confidence")
    # Inset placement & size as fractions of the frame
    ap.add_argument("--inset-w-frac", type=float, default=0.28, help="Inset width as fraction of frame width")
    ap.add_argument("--inset-h-frac", type=float, default=0.28, help="Inset height as fraction of frame height")
    ap.add_argument("--inset-margin", type=int,   default=12,   help="Inset margin (pixels) from top-right corner")
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

            # Draw results on the live hand (green)
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

                # Build the 42-D canonical vector for the FIRST hand (for the inset)
                h, w = img.shape[:2]
                lm0 = res.multi_hand_landmarks[0]
                pts = np.array([(p.x * w, p.y * h) for p in lm0.landmark], dtype=np.float32)
                handed0 = (res.multi_handedness[0].classification[0].label
                           if res.multi_handedness else None)
                canon_vec = normalize_landmarks(pts, handed0)

                # Compute inset position/size
                inset_w = int(args.inset_w_frac * w)
                inset_h = int(args.inset_h_frac * h)
                x0 = w - inset_w - args.inset_margin
                y0 = args.inset_margin

                # Draw canonicalized landmarks in magenta inset
                draw_canonical_inset(
                    img,
                    canon_vec,
                    inset_anchor=(x0, y0),
                    inset_size=(inset_w, inset_h),
                    color=(255, 0, 255),
                    box_color=(255, 0, 255),
                    label="Canonical (preprocessed)"
                )

            # FPS
            t = time.time()
            dt = max(t - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # EMA smoothing
            t_prev = t
            put_text(img, f"FPS: {fps:.1f}   (ESC to quit)", (10, 30), 0.8, (255, 255, 255), 2)

            cv2.imshow("MediaPipe Hands — Landmarks + Canonical Inset", img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
