import unittest
import utils
import cv2
import cProfile

resize_to_height = 1536
image = cv2.imread('./sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (round(image.shape[0] * (resize_to_height / image.shape[1])), resize_to_height))
image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

class UtilsTest(unittest.TestCase):

    def test_get_connected_components(self):

        cProfile.run('utils.get_connected_components(image)')

if __name__ == '__main__':
    unittest.main()
