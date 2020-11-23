import os
from aipollo_omr.detectors import unet_torch
import torch


class ClefDetector:

    def __init__(self, models_dir):
        self._nn = unet_torch.models.UNet()
        self._nn.load_state_dict(
            torch.load(os.path.join(models_dir, 'treble_clefs.pt')))
        self._nn.eval()
        torch.no_grad()

    def detect(self, image):
        raise NotImplementedError