from aipollo_omr.detectors import unet_torch
from . import utils
import torch


class ClefDetector:

    def __init__(self):
        self._nn = unet_torch.models.UNet()
        self._nn.load_state_dict(
            torch.load(
                str(utils.MODELS_DIR /
                    'aipollo_omr/detectors/unet_torch/logs/[9]--2020-10-29-14.12.51/15000.pt'
                   )))
        self._nn.eval()
        torch.no_grad()

    def detect(self, image):
        raise NotImplementedError