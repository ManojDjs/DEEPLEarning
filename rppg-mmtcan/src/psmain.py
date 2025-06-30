# backend.py
import cv2
import torch
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from torchvision import transforms
from scipy.signal import periodogram
from models.psnet_model import PhysNet_padding_Encoder_Decoder_MAX

FRAME_DEPTH = 128
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'models/pre_trained_models/PURE_PhysNet_DiffNormalized.pth'

app = FastAPI()

model = PhysNet_padding_Encoder_Decoder_MAX(frames=FRAME_DEPTH).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

frame_buffer = deque(maxlen=FRAME_DEPTH)

def preprocess_face(frame):
    # frame is expected as numpy RGB image, crop/resize as needed (assuming full frame face)
    face_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    tensor = transform(face_resized)
    return tensor

def estimate_hr(rppg_signal, fs=30):
    rppg = rppg_signal.cpu().numpy().flatten()
    f, pxx = periodogram(rppg, fs=fs)
    valid = (f >= 0.7) & (f <= 2.5)
    if not np.any(valid):
        return None
    peak_freq = f[valid][np.argmax(pxx[valid])]
    bpm = peak_freq * 60.0
    return bpm

@app.get("/")
async def get():
    html_content = open("static/frontend.html").read()
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_buffer.clear()

    try:
        while True:
            # Receive base64 image string from client
            data = await websocket.receive_text()

            # Decode base64 to np.array
            import base64
            import io
            from PIL import Image

            img_data = base64.b64decode(data.split(",")[1])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            frame_np = np.array(img)

            # Preprocess
            face_tensor = preprocess_face(frame_np)
            frame_buffer.append(face_tensor)

            bpm = None
            if len(frame_buffer) == FRAME_DEPTH:
                input_tensor = torch.stack(list(frame_buffer), dim=1)  # (3, T, H, W)
                input_tensor = input_tensor.unsqueeze(0).to(DEVICE)   # (1, 3, T, H, W)

                with torch.no_grad():
                    rppg_signal, *_ = model(input_tensor)

                bpm = estimate_hr(rppg_signal)

            # Send back bpm or None
            await websocket.send_json({"bpm": bpm})

    except WebSocketDisconnect:
        print("Client disconnected")
