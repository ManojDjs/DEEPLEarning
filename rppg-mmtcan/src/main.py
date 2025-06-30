import cv2
import torch
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from collections import deque
from time import time
import base64
import io
import os
from .models.deepphys_model import DeepPhys as DP
# from .models.deepphys_model import DeepPhys as DP
from PIL import Image
import logging

# Ensure the logger is set up to capture debug information
logging.basicConfig(level=logging.INFO)

logger= logging.getLogger(__name__)


import torch.nn as nn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
STATIC_PATH = os.path.join(os.getcwd(), 'src/static')
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 72

class Attention_mask(nn.Module):
    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

class DeepPhys(nn.Module):
    def __init__(self, img_size=72):
        super(DeepPhys, self).__init__()
        # same model as before...
        self.in_channels = 3
        self.kernel_size = 3
        self.pool_size = (2, 2)
        self.nb_filters1 = 32
        self.nb_filters2 = 64
        self.nb_dense = 128
        self.dropout_rate1 = 0.25
        self.dropout_rate2 = 0.5

        self.motion_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.motion_conv2 = nn.Conv2d(32, 32, 3)
        self.motion_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.motion_conv4 = nn.Conv2d(64, 64, 3)

        self.appearance_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.appearance_conv2 = nn.Conv2d(32, 32, 3)
        self.appearance_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.appearance_conv4 = nn.Conv2d(64, 64, 3)

        self.att_conv1 = nn.Conv2d(32, 1, 1)
        self.att_conv2 = nn.Conv2d(64, 1, 1)
        self.attn_mask_1 = Attention_mask()
        self.attn_mask_2 = Attention_mask()

        self.pool = nn.AvgPool2d(self.pool_size)
        self.dropout1 = nn.Dropout(self.dropout_rate1)
        self.dropout2 = nn.Dropout(self.dropout_rate1)
        self.dropout3 = nn.Dropout(self.dropout_rate1)
        self.dropout4 = nn.Dropout(self.dropout_rate2)

        if img_size == 72:
            self.fc1 = nn.Linear(16384, self.nb_dense)
        elif img_size == 36:
            self.fc1 = nn.Linear(3136, self.nb_dense)
        elif img_size == 96:
            self.fc1 = nn.Linear(30976, self.nb_dense)
        else:
            raise Exception("Unsupported image size")

        self.fc2 = nn.Linear(self.nb_dense, 1)

    def forward(self, x):
        diff = x[:, :3, :, :]
        raw = x[:, 3:, :, :]

        d1 = torch.tanh(self.motion_conv1(diff))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.appearance_conv1(raw))
        r2 = torch.tanh(self.appearance_conv2(r1))

        g1 = torch.sigmoid(self.att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.pool(gated1)
        d4 = self.dropout1(d3)

        r3 = self.pool(r2)
        r4 = self.dropout2(r3)

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.appearance_conv3(r4))
        r6 = torch.tanh(self.appearance_conv4(r5))

        g2 = torch.sigmoid(self.att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.pool(gated2)
        d8 = self.dropout3(d7)
        d9 = d8.view(d8.size(0), -1)

        d10 = torch.tanh(self.fc1(d9))
        d11 = self.dropout4(d10)
        out = self.fc2(d11)

        return torch.sigmoid(out) * 200

model = DeepPhys(img_size=IMG_SIZE).to(DEVICE)
model.eval()

frame_buffer = deque(maxlen=2)
last_predict_time = 0

EVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 72

# Define your DeepPhys class here (as before) ...

model = DeepPhys(img_size=IMG_SIZE).to(DEVICE)

# =================== Parameters ===================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 72
FRAME_DEPTH = 20

# =================== Model ========================
model = DP(img_size=IMG_SIZE).to(DEVICE)
MODEL_PATH = os.path.join(os.getcwd(), 'src/models/BP4D_PseudoLabel_DeepPhys.pth')
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# state_dict = torch.load('BP4D_PseudoLabel_DeepPhys.pth', map_location=DEVICE)x
# state_dict = torch.load('path_to_checkpoint.pth', map_location=DEVICE)

state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()
import os
# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG to see all messages
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log the current working directory
cwd = os.getcwd()
logger.info(f"Current working directory: {cwd}")

# List and log files/folders in current directory
entries = os.listdir(cwd)
logger.info(f"Files and folders in {cwd}: {entries}")
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    global frame_buffer
    global last_predict_time

    try:
        while True:
            data = await websocket.receive_text()
            # data is base64 image string from frontend

            # Remove "data:image/jpeg;base64," prefix
            if data.startswith("data:image/jpeg;base64,"):
                data = data[len("data:image/jpeg;base64,"):]

            img_bytes = base64.b64decode(data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_np = np.array(img)

            rgb_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0  # (3, H, W)
            frame_buffer.append(rgb_tensor)

            current_time = time()
            if current_time - last_predict_time >= 3 and len(frame_buffer) == 2:
                diff = frame_buffer[1] - frame_buffer[0]
                combined = torch.cat([diff, frame_buffer[1]], dim=0).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    hr = model(combined).item()

                await websocket.send_json({"heart_rate": round(hr, 2)})
                last_predict_time = current_time

    except Exception as e:
        print(f"WebSocket connection closed: {e}")
