import torch.nn as nn


class DiffusionNetwork(nn.Module):
    def __init__(self):
        super(DiffusionNetwork, self).__init__()

        # Down sampling layers
        self.down_sample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Up sampling layers
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Upscale layer for compatibility with 900x900 images
        self.upscale = nn.Upsample(size=(900, 900), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Upscale input to 900x900
        x = self.upscale(x)

        # Apply down sampling
        down_sampled = self.down_sample(x)

        # Apply up sampling
        up_sampled = self.up_sample(down_sampled)

        return up_sampled
