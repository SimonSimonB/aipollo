import unittest
import cv2
import staff_line_detector 

class StaffLineDetectorTest(unittest.TestCase):

    def test1(self):
        resize_to_height = 1024
        image = cv2.imread('./sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
        image = cv2.imread('./sample_scans/bleib.jpg', cv2.IMREAD_GRAYSCALE)
        image = cv2.imread('./sample_scans/be_still.jpg', cv2.IMREAD_GRAYSCALE)
        image = cv2.imread('./sample_scans/brown.jpg', cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (round(image.shape[1] * (resize_to_height / image.shape[0])), resize_to_height))
        detector = staff_line_detector.StaffLineDetector()
        detected = detector.detect(image)

if __name__ == '__main__':
    unittest.main()