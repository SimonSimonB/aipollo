import cv2
#import ScoreModel
from aipollo_processor.detectors.unet_torch import unet_torch_manager
from aipollo_processor.detectors.staff_detector import StaffDetector
from aipollo_processor.detectors.note_detector import NoteDetector

object_detectors = []

def analyze_score(image):
    '''Returns 
    '''

    all_score_elements = []

    print('Start detecting staffs...')
    staffs, staff_height = StaffDetector().detect(image)
    all_score_elements.extend(staffs)
    print('Done.')
    print('Start detecting half notes...')
    all_score_elements.extend(NoteDetector().detect(image, staff_height))
    print('Done.')

    return all_score_elements

if __name__ == '__main__':
    image = cv2.imread('../sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
    analyze_score(image)
    #unet_torch_manager.train_net([[36]])
