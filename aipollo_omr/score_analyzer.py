from aipollo_omr.detectors.staff_detector import StaffDetector
from aipollo_omr.detectors.note_detector import NoteDetector

object_detectors = []


def analyze_score(image, models_dir):
    all_score_elements = []

    print('Start detecting staffs...')
    staffs, staff_height = StaffDetector(models_dir).detect(image)
    all_score_elements.extend(staffs)
    print('Done.')
    print('Start detecting half notes...')
    all_score_elements.extend(
        NoteDetector(models_dir).detect(image, staff_height))
    print('Done.')

    return all_score_elements