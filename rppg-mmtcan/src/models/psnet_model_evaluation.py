import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import deque
from scipy.signal import periodogram
from psnet_model import PhysNet_padding_Encoder_Decoder_MAX  # Assuming your model is saved here

# Parameters
FRAME_DEPTH = 128
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'pre_trained_models/PURE_PhysNet_DiffNormalized.pth'  # Path to your trained weights
# Load Model
model = PhysNet_padding_Encoder_Decoder_MAX(frames=FRAME_DEPTH).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Path to your trained model
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),  # [HWC] to [CHW], 0-255 to 0.0-1.0
])

# Face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Frame buffer
frame_buffer = deque(maxlen=FRAME_DEPTH)

def preprocess_face(frame, box):
    x, y, w, h = box
    roi = frame[y:y+h, x:x+w]
    face_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    tensor = transform(face_resized)  # (3, H, W)
    return tensor

def estimate_hr(rppg_signal, fs=30):
    rppg = rppg_signal.cpu().numpy().flatten()
    f, pxx = periodogram(rppg, fs=fs)
    valid = (f >= 0.7) & (f <= 2.5)  # approx 42 bpm to 150 bpm
    if not np.any(valid):
        return None
    peak_freq = f[valid][np.argmax(pxx[valid])]
    bpm = peak_freq * 60.0
    return bpm

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

print("[INFO] Starting real-time heart rate estimation...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_tensor = preprocess_face(frame_rgb, (x, y, w, h))
            frame_buffer.append(face_tensor)

            if len(frame_buffer) == FRAME_DEPTH:
                input_tensor = torch.stack(list(frame_buffer), dim=1)  # (3, T, H, W)
                input_tensor = input_tensor.unsqueeze(0).to(DEVICE)   # (1, 3, T, H, W)
                with torch.no_grad():
                    rppg_signal, *_ = model(input_tensor)
                bpm = estimate_hr(rppg_signal)

                if bpm is not None:
                    text = f"Heart Rate: {bpm:.1f} bpm"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("PhysNet Heart Rate Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
