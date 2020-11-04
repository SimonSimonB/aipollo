import unittest
import cv2
import staff_detector 

class StaffDetectorTest(unittest.TestCase):

    def test_bleib_rotated(self):
        image = cv2.imread('./sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
        detector = staff_detector.StaffDetector()
        staffs = detector.detect(image)
        self.assertEqual(8, len(staffs))


    def test_brown(self):
        image = cv2.imread('./sample_scans/brown.jpg', cv2.IMREAD_GRAYSCALE)
        detector = staff_detector.StaffDetector()
        staffs = detector.detect(image)
        self.assertEqual(14, len(staffs))

    def test_be_still(self):
        image = cv2.imread('./sample_scans/be_still.jpg', cv2.IMREAD_GRAYSCALE)
        detector = staff_detector.StaffDetector()
        staffs = detector.detect(image)
        self.assertEqual(8, len(staffs))

if __name__ == '__main__':
    unittest.main()