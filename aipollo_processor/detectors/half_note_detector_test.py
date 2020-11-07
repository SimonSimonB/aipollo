import unittest
from .half_note_detector import HalfNoteDetector
import cv2

class HalfNoteDetectorTest(unittest.TestCase):
    def test_bleib(self):
        image = cv2.imread('./sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
        hnd = HalfNoteDetector()

        hnd.detect(image, 30)

if __name__ == '__main__':
    unittest.main()