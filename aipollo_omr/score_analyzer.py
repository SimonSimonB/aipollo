from aipollo_omr.detectors.staff_detector import StaffDetector
from aipollo_omr.detectors.note_detector import NoteDetector

object_detectors = []


def analyze_score(image):
    all_score_elements = []

    print('Start detecting staffs...')
    staffs, staff_height = StaffDetector().detect(image)
    all_score_elements.extend(staffs)
    print('Done.')
    print('Start detecting half notes...')
    all_score_elements.extend(NoteDetector().detect(image, staff_height))
    print('Done.')

    return all_score_elements