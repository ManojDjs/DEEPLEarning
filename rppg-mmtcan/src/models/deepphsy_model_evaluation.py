import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
import torch.nn as nn

from deepphys_model import DeepPhys  # Adjust the import based on your project structure



# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepPhys(img_size=72).to(device)
state_dict=torch.load('BP4D_PseudoLabel_DeepPhys.pth', map_location=device)  # Path to your pretrained model
state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()

# Initialize OpenCV
cap = cv2.VideoCapture(0)  # Use your webcam

# Initialize variables
prev_frame = None
heart_rate_list = []
time_list = []
start_time = time()
prev_time = time()
heart_rate_bpm = 0.0  # <-- Initialize here

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Live plot setup
plt.ion()
fig, ax = plt.subplots()
ax.set_title('Live Heart Rate')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Heart Rate (BPM)')
line, = ax.plot([], [], 'r-')

# Function to update graph
def update_graph():
    line.set_xdata(time_list)
    line.set_ydata(heart_rate_list)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.1)

# Start live video feed and prediction
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Only proceed if a face is detected
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Use the first detected face
        forehead = frame[y:y + h // 3, x:x + w]
        forehead_resized = cv2.resize(forehead, (72, 72))

        # Compute diff frame
        if prev_frame is not None:
            diff = cv2.absdiff(forehead_resized, prev_frame)

            # Prepare input
            raw_rgb = np.transpose(forehead_resized, (2, 0, 1))
            motion_rgb = np.transpose(diff, (2, 0, 1))
            input_tensor = np.concatenate([motion_rgb, raw_rgb], axis=0)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            # Check time to infer
            current_time = time()
            if current_time - prev_time > 2:
                prev_time = current_time

                with torch.no_grad():
                    heart_rate = model(input_tensor)
                    heart_rate_bpm = heart_rate.item()

                # Update graph
                heart_rate_list.append(heart_rate_bpm)
                time_list.append(current_time - start_time)
                update_graph()

        # Update prev_frame after processing
        prev_frame = forehead_resized.copy()

        # Display heart rate
        cv2.putText(frame, f"Heart Rate: {heart_rate_bpm:.2f} BPM", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video
    cv2.imshow('Live Feed', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Clean up
cap.release()
cv2.destroyAllWindows()
plt.ioff()
