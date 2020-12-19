import unittest
import cv2
from . import note_detector


class NoteDetectorTest(unittest.TestCase):

    def test_bleib_rotated(self):
        image = cv2.imread('./sample_scans/bleib_rotated.jpg',
                           cv2.IMREAD_GRAYSCALE)
        detector = note_detector.NoteDetector(
            r'C:/Users/simon/Coding/ML/aipollo/aipollo_omr/detectors/unet_torch/models'
        )
        notes = detector.detect(image, 30)
        self.assertTrue(len(notes) > 150 and len(notes) < 250)


if __name__ == '__main__':
    unittest.main()