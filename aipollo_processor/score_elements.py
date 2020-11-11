from typing import List
from aipollo_processor.detectors import geometry_utils 
import collections
import enum

'''
StaffLine = collections.namedtuple('StaffLine', 'which points_along')
Accidental = collections.namedtuple('Accidental', 'type center_position')
Note = collections.namedtuple('Note', 'center_position bounding_box pitch duration')
'''


class ScoreElementType(enum.Enum):
    staff = 0
    half_note = 1
    staff_line = 2

class ScoreElement:
    def __init__(self, element_type: ScoreElementType, pixels: List[geometry_utils.Point]):
        self.type = element_type
        self.pixels = pixels
    
    def bounding_box(self):
        return geometry_utils.get_bounding_box(self.pixels)
