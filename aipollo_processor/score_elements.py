from typing import List
from aipollo_processor.detectors.geometry_utils import Point
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

class ScoreElement:
    def __init__(self, element_type: ScoreElementType, pixels: List[Point]):
        self.type = element_type
        self.pixels = pixels
