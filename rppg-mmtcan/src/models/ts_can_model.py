import torch
import torch.nn as nn


class AttentionMask(nn.Module):
    """Spatial Attention Mask for feature modulation."""

    def forward(self, x):
        xsum = torch.sum(x, dim=(2, 3), keepdim=True)
        return x / xsum * x.size(2) * x.size(3) * 0.5


class TSM(nn.Module):
    """Temporal Shift Module."""

    def __init__(self, n_segment=10, fold_div=3):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]          # shift left
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift right
        out[:, :, 2*fold:] = x[:, :, 2*fold:]         # static
        return out.view(nt, c, h, w)


class BaseTSCAN(nn.Module):
    """Base class with shared logic between TSCAN and MTTS-CAN."""

    def __init__(self, in_channels, nb_filters1, nb_filters2, kernel_size, pool_size, dropout_rate1, dropout_rate2, frame_depth):
        super().__init__()
        self.TSM = nn.ModuleList([TSM(n_segment=frame_depth) for _ in range(4)])

        self.motion_convs = nn.ModuleList([
            nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1),
            nn.Conv2d(nb_filters1, nb_filters1, kernel_size),
            nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1),
            nn.Conv2d(nb_filters2, nb_filters2, kernel_size)
        ])

        self.appearance_convs = nn.ModuleList([
            nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1),
            nn.Conv2d(nb_filters1, nb_filters1, kernel_size),
            nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1),
            nn.Conv2d(nb_filters2, nb_filters2, kernel_size)
        ])

        self.attention_convs = nn.ModuleList([
            nn.Conv2d(nb_filters1, 1, kernel_size=1),
            nn.Conv2d(nb_filters2, 1, kernel_size=1)
        ])
        self.attn_masks = nn.ModuleList([AttentionMask(), AttentionMask()])

        self.pooling = nn.ModuleList([
            nn.AvgPool2d(pool_size),
            nn.AvgPool2d(pool_size),
            nn.AvgPool2d(pool_size)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate1),
            nn.Dropout(dropout_rate1),
            nn.Dropout(dropout_rate1)
        ])

    def _shared_forward(self, diff_input, raw_input):
        # Stage 1
        d = torch.tanh(self.motion_convs[0](self.TSM[0](diff_input)))
        d = torch.tanh(self.motion_convs[1](self.TSM[1](d)))

        r = torch.tanh(self.appearance_convs[0](raw_input))
        r = torch.tanh(self.appearance_convs[1](r))

        g = torch.sigmoid(self.attention_convs[0](r))
        g = self.attn_masks[0](g)
        gated = d * g

        d = self.dropouts[0](self.pooling[0](gated))
        r = self.dropouts[1](self.pooling[1](r))

        # Stage 2
        d = torch.tanh(self.motion_convs[2](self.TSM[2](d)))
        d = torch.tanh(self.motion_convs[3](self.TSM[3](d)))

        r = torch.tanh(self.appearance_convs[2](r))
        r = torch.tanh(self.appearance_convs[3](r))

        g = torch.sigmoid(self.attention_convs[1](r))
        g = self.attn_masks[1](g)
        gated = d * g

        d = self.dropouts[2](self.pooling[2](gated))
        return d.view(d.size(0), -1)


class TSCAN(BaseTSCAN):
    """Temporal Shift Convolutional Attention Network (TS-CAN) for heart rate estimation."""

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3,
                 dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128,
                 frame_depth=20, img_size=36):
        super().__init__(in_channels, nb_filters1, nb_filters2, kernel_size,
                         pool_size, dropout_rate1, dropout_rate2, frame_depth)

        input_dims = {
            36: 3136,
            72: 16384,
            96: 30976,
            128: 57600
        }

        if img_size not in input_dims:
            raise ValueError(f"Unsupported image size: {img_size}")

        self.fc = nn.Sequential(
            nn.Linear(input_dims[img_size], nb_dense),
            nn.Tanh(),
            nn.Dropout(dropout_rate2),
            nn.Linear(nb_dense, 1)
        )

    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3]
        raw_input = inputs[:, 3:]
        features = self._shared_forward(diff_input, raw_input)
        return self.fc(features)


class MTTS_CAN(BaseTSCAN):
    """Multi-task TS-CAN for both heart rate and respiration estimation."""

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3,
                 dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128,
                 frame_depth=20):
        super().__init__(in_channels, nb_filters1, nb_filters2, kernel_size,
                         pool_size, dropout_rate1, dropout_rate2, frame_depth)

        self.fc_hr = nn.Sequential(
            nn.Linear(16384, nb_dense),
            nn.Tanh(),
            nn.Dropout(dropout_rate2),
            nn.Linear(nb_dense, 1)
        )

        self.fc_resp = nn.Sequential(
            nn.Linear(16384, nb_dense),
            nn.Tanh(),
            nn.Dropout(dropout_rate2),
            nn.Linear(nb_dense, 1)
        )

    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3]
        raw_input = inputs[:, 3:]
        features = self._shared_forward(diff_input, raw_input)
        return self.fc_hr(features), self.fc_resp(features)
