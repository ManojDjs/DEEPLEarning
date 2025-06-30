import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from ts_can_model import MTTS_CAN
import os

# ================== Parameters ==================
FRAME_DEPTH = 20
IMG_SIZE = 36
MODEL_PATH = 'pre_trained_models/PURE_TSCAN.pth'  # Path to your trained weights
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================== Load Model ==================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please ensure the file exists.")

model = MTTS_CAN().to(DEVICE)

# Load checkpoint and strip 'module.' prefix if present
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
if any(k.startswith('module.') for k in state_dict.keys()):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    state_dict = new_state_dict

model.load_state_dict(state_dict, strict=False)
model.eval()

# ================== Utilities ==================
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts HWC [0,255] to CHW [0.0,1.0]
])

def preprocess_frame(frame, size):
    face = cv2.resize(frame, (size, size))
    return transform(face).unsqueeze(0)  # (1, 3, H, W)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Buffers
raw_buffer = []
diff_buffer = []

# ================== Start Webcam ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("[INFO] Starting webcam stream...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Use first detected face
            roi = frame_rgb[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            preprocessed = preprocess_frame(roi, IMG_SIZE)
            raw_buffer.append(preprocessed)

            if len(raw_buffer) > 1:
                diff = raw_buffer[-1] - raw_buffer[-2]
                diff_buffer.append(diff)

            # We need at least FRAME_DEPTH frames and FRAME_DEPTH - 1 diffs
            if len(raw_buffer) >= FRAME_DEPTH and len(diff_buffer) >= FRAME_DEPTH - 1:
                raw_seq = torch.cat(raw_buffer[-FRAME_DEPTH:], dim=0)  # (F, 3, H, W)
                diff_seq = torch.cat(diff_buffer[-(FRAME_DEPTH - 1):], dim=0)  # (F-1, 3, H, W)

                # Ensure the input tensor has the correct shape
                if raw_seq.shape[0] == FRAME_DEPTH and diff_seq.shape[0] == FRAME_DEPTH - 1:
                    input_tensor = torch.cat([diff_seq, raw_seq[:-1]], dim=1).unsqueeze(0).to(DEVICE)  # (1, F-1, 6, H, W)

                    # Reshape input tensor to match model's expected input
                    input_tensor = input_tensor.permute(0, 2, 1, 3, 4)  # (1, 6, F-1, H, W)

                    with torch.no_grad():
                        hr_pred, rr_pred = model(input_tensor)

                    heart_rate = hr_pred.item()
                    respiration_rate = rr_pred.item()

                    # Display the rates
                    text = f"HR: {heart_rate:.2f} bpm | RR: {respiration_rate:.2f} bpm"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("MTTS-CAN Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print
