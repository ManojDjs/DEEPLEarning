# models/mmt_can.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.fc2 = nn.Linear(in_channels // 8, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)
        return x * y

class MMT_CAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Conv3DBlock(3, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.att1 = AttentionBlock(32)
        self.block2 = Conv3DBlock(32, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.att2 = AttentionBlock(64)
        self.final_conv = nn.Conv3d(64, 1, (1, 1, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.att1(x)
        x = self.block2(x)
        x = self.att2(x)
        x = self.final_conv(x)
        return x.view(x.size(0), -1)

    def infer(self, roi_sequence):
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(roi_sequence).float().permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
            return self.forward(input_tensor).cpu().numpy()

    def load_pretrained(self, path="mmt_can_weights.pth"):
        self.load_state_dict(torch.load(path, map_location="cpu"))
