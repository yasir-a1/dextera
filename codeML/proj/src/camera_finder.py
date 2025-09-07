import cv2, time

def scan(max_idx=10, backend=None):
    hits = []
    for i in range(max_idx+1):
        cap = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
        if cap.isOpened():
            # try to grab one frame
            time.sleep(0.1)
            ok, _ = cap.read()
            if ok:
                hits.append(i)
        cap.release()
    return hits

# Try both Windows backends; some cameras work on one but not the other
print("DirectShow:", scan(10, cv2.CAP_DSHOW))
print("Media Foundation:", scan(10, cv2.CAP_MSMF))
