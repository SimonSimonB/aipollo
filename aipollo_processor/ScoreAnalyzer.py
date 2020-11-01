from aipollo_processor.ScoreModel import ScoreModel


object_detectors = []

def analyze_score(image):
    score_elements = []
    for object_detector in object_detectors:
        object_detector.detect(image)

    score_model = ScoreModel(score_elements)

    return score_model