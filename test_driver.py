from aipollo_processor.detectors import staff_detector_test
from aipollo_processor import score_analyzer
import cv2

image = cv2.imread('./sample_scans/be_still.jpg', cv2.IMREAD_GRAYSCALE)
t = score_analyzer.analyze_score(image)

#t = staff_detector_test.StaffDetectorTest()
#t.test_be_still()
#t.test_be_still()