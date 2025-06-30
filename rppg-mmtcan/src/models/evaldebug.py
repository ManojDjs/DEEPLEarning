import cv2
import numpy as np
import torch
import time
from deepphys_model import DeepPhys  # Adjust the import based on your project structure
# Dummy CNN model (replace with your own model)
model = DeepPhys()  # Your model
model.to('cpu')  # Ensure model is on CPU

# Load model with checkpoint
checkpoint = torch.load("BP4D_PseudoLabel_DeepPhys.pth", map_location=torch.device('cpu'))

# Fix the state_dict key names
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in checkpoint.items():
    name = k.replace("module.", "")  # remove `module.` prefix
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

print("Model loaded successfully.",model.eval())

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Start camera
cap = cv2.VideoCapture(0)

start_time = time.time()
last_infer_time = 0
ready_to_predict = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    elapsed_time = time.time() - start_time

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    forehead_rgb = None
    for (x, y, w, h) in faces:
        fx = x + int(w * 0.3)
        fy = y + int(h * 0.15)
        fw = int(w * 0.4)
        fh = int(h * 0.15)

        # Ensure the forehead region is within frame bounds
        if fy >= 0 and fx >= 0 and fy+fh <= frame.shape[0] and fx+fw <= frame.shape[1]:
            forehead_rgb = frame[fy:fy+fh, fx:fx+fw]
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
        break

    # Display RGB channel values
    if forehead_rgb is not None and forehead_rgb.size != 0 and forehead_rgb.shape[2] == 3:
        avg_rgb = np.mean(forehead_rgb, axis=(0, 1))  # BGR
        r, g, b = avg_rgb[2], avg_rgb[1], avg_rgb[0]  # Convert to RGB
        cv2.putText(frame, f"R: {r:.2f} G: {g:.2f} B: {b:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Allow start after 'o' key or 10s wait
    if key == ord('o') or elapsed_time > 5:
        ready_to_predict = True

    # Run inference every 2 seconds
    if (ready_to_predict and forehead_rgb is not None and
        forehead_rgb.size != 0 and
        len(forehead_rgb.shape) == 3 and
        forehead_rgb.shape[2] == 3 and
        time.time() - last_infer_time > 2):
        try:
            # Preprocess
            forehead_resized = cv2.resize(forehead_rgb, (128, 128))
            forehead_tensor = np.transpose(forehead_resized, (2, 0, 1))  # (C, H, W)
            forehead_tensor = forehead_tensor / 255.0  # Normalize to [0,1]
            input_tensor = torch.tensor(forehead_tensor, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
            input_tensor = input_tensor.float()  # Ensure float32
            input_tensor = input_tensor.to('cpu')  # Ensure tensor is on CPU

            # Predict
            with torch.no_grad():
                prediction = model(input_tensor)
                heart_rate = prediction.item()

            # Display HR
            cv2.putText(frame, f"HR: {heart_rate:.2f} bpm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            last_infer_time = time.time()

        except Exception as e:
            print(f"[!] Model error: {e}")
            cv2.putText(frame, "Model Error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Forehead Feed", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
