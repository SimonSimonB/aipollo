from aipollo_processor.detectors.utils import draw_score_annotations
from aipollo_processor.detectors.geometry_utils import Point
from aipollo_processor.score_elements import ScoreElement, ScoreElementType
import unittest
from aipollo_processor.detectors import utils
import cv2
import cProfile

resize_to_height = 1536
image = cv2.imread('./sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (round(image.shape[0] * (resize_to_height / image.shape[1])), resize_to_height))
image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

class UtilsTest(unittest.TestCase):

    def test_get_connected_components(self):

        cProfile.run('utils.get_connected_components(image)')
    
    def test_draw_score_annotations(self):
        image = cv2.imread('./sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
        annotations = [ScoreElement(ScoreElementType.staff_line, [Point(y, x) for y, x in zip(range(30), range(30))])]
        draw_score_annotations(image, annotations)

if __name__ == '__main__':
    unittest.main()
