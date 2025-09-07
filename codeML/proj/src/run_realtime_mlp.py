# run_realtime_mlp.py
import time, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import cv2, mediapipe as mp
import serial as serial

MODEL_DIR = Path("models")
CLASSES   = [c.strip() for c in open(MODEL_DIR/"classes.txt").read().splitlines()]
last_pose = "NO_HAND"

class MLP(nn.Module):
    def __init__(self, in_dim=42, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(32, n_classes),
        )
    def forward(self, x): return self.net(x)

def _angle(a, b):
    return math.atan2(a[1]-b[1], a[0]-b[0])

def normalize_landmarks(lm_xy, handedness_label=None):
    pts = np.array(lm_xy, dtype=np.float32)
    if handedness_label and handedness_label.lower().startswith('l'):
        pts[:,0] = -pts[:,0]
    wrist = pts[0].copy()
    pts -= wrist
    axis = pts[9]
    ang  = _angle(axis, np.array([0.0,0.0], dtype=np.float32))
    ca, sa = math.cos(-ang), math.sin(-ang)
    R = np.array([[ca,-sa],[sa,ca]], dtype=np.float32)
    pts = pts @ R.T
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.reshape(-1)

def sendUART(confidence: float, label: str, serialCommObject: serial.Serial):
    global last_pose
    
    if (confidence >= 0.87) and (label != last_pose):
        last_pose = label
        print(last_pose)
        msg = f"{last_pose}\n"
        serialCommObject.write(msg.encode("ascii"))
    

def main():
    #serialComm = serial.Serial('COM5',115200) #change for differnet COM port
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=42, n_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_DIR/"gesture_mlp.pt", map_location=device))
    model.eval()

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    prob_smooth = None
    alpha = 0.4  # EMA smoothing

    with mp_hands.Hands(static_image_mode=False, model_complexity=0, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            img = cv2.flip(frame, 1)
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            label, conf = "NO_HAND", 1.0
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                pts = [(p.x*w, p.y*h) for p in lm.landmark]
                handed = res.multi_handedness[0].classification[0].label if res.multi_handedness else None
                feat = normalize_landmarks(pts, handed)
                x = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                # EMA
                if prob_smooth is None:
                    prob_smooth = probs
                else:
                    prob_smooth = (1-alpha)*prob_smooth + alpha*probs
                idx = int(np.argmax(prob_smooth))
                label = CLASSES[idx]
                conf  = float(prob_smooth[idx])

                #addition of script confidence and printout for UART readout
            

                mp_draw.draw_landmarks(
                    img, lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style())
            
            #sendUART(conf, label=label, serialCommObject=serialComm)

            cv2.putText(img, f"Gesture: {label} ({conf:.2f})", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Gesture MLP — Realtime", img)
            if cv2.waitKey(1) & 0xFF == 27:
                #serialComm.close()
                break


    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
