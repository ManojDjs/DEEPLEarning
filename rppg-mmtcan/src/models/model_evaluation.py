import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
import torch.nn as nn

# DeepPhys model definition
class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

class DeepPhys(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36):
        super(DeepPhys, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Appearance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)
        out = torch.sigmoid(out) * 200  # Assuming max HR is 200
        return out


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
