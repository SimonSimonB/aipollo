#import ScoreModel
from detectors.unet_torch import unet_torch_manager
import score_model


object_detectors = []

def analyze_score(image):
    score_elements = []
    for object_detector in object_detectors:
        object_detector.detect(image)

    score_model = score_model.ScoreModel(score_elements)

    return score_model

if __name__ == '__main__':
    unet_torch_manager.train_net([[9]])
