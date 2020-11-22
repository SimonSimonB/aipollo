from typing import List
from aipollo_omr.detectors import geometry_utils
import enum


class ScoreElementType(enum.Enum):
    staff = 0
    half_note = 1
    quarter_note = 2
    staff_line = 3


class ScoreElement:

    def __init__(self,
                 element_type: ScoreElementType,
                 pixels: List[geometry_utils.Point],
                 children: List[ScoreElementType] = None):
        self.type = element_type
        self.pixels = pixels
        self.children = [] if children == None else children

    def bounding_box(self):
        return geometry_utils.get_bounding_box(self.pixels)
