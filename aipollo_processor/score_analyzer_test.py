import cv2
import unittest
from aipollo_processor import score_analyzer

class ScoreAnalyzerTest(unittest.TestCase):
    def test_case(self):
        image = cv2.imread('./sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
        score_analyzer.analyze_score(image)
