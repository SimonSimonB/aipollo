import unittest
import cv2
import clef_detector 

class ClefDetectorTest(unittest.TestCase):

    def test1(self):
        resize_to_height = 1600
        image = cv2.imread('./sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (round(image.shape[1] * (resize_to_height / image.shape[0])), resize_to_height))
        detector = clef_detector.ClefDetector()
        detected = detector.detect(image)
        a = 5

if __name__ == '__main__':
    unittest.main()