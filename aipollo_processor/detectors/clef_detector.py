import unet_torch.models
import utils
import torch

class ClefDetector:
    
    def __init__(self) -> None:
        self._nn = unet_torch.models.UNet()
        self._nn.load_state_dict(torch.load('aipollo_processor/detectors/unet_torch/logs/[9]--2020-10-29-14.12.51/15000.pt'))
        self._nn.eval()
        torch.no_grad()
    
    def detect(self, image):
        mask = utils.classify(image, self._nn)
        utils.show(image)
        utils.show(mask)