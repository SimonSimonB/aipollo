import os

from numpy.core.fromnumeric import resize
from aipollo_processor.score_elements import ScoreElement, ScoreElementType
from . import utils
from .unet_torch import models
import torch
import cv2
from . import geometry_utils 
from .geometry_utils import Point

class HalfNoteDetector:

    def __init__(self):
        self._nn = models.UNet()
        self._nn.load_state_dict(torch.load(os.path.join(utils.MODELS_DIR, r'[36]--2020-11-05-15.07.33/3000.pt')))
        self._nn.eval()
        torch.no_grad()

        self._detection_friendly_staff_height = 30

    def detect(self, image, staff_height):
        # Rescale image so that staff height equals detection-friendly staff height.
        resize_factor = self._detection_friendly_staff_height / staff_height
        image = cv2.resize(image, (round(image.shape[1] * resize_factor), round(image.shape[0] * resize_factor)))

        # Detect half notes.
        half_notes = self._detect(image, staff_height)

        # Resize them to the original height of the image.
        half_notes = [
            ScoreElement(
                ScoreElementType.half_note, 
                [(1 / resize_factor) * point for point in half_note.pixels]
            ) for half_note in half_notes
        ]

        print(f'Found {len(half_notes)} half notes.')

        return half_notes


    def _detect(self, image, staff_height):
        utils.show(image)
        # Preprocess image (reshaping etc.)
        mask = utils.classify(image, self._nn)
        #utils.show(image)
        #utils.show(mask)

        # Threshold the pixel-wise classification
        threshold = 0.1
        mask[mask > threshold] = 1.0
        mask[mask <= threshold] = 0.0
        utils.show(mask)

        connected_components = geometry_utils.get_connected_components(mask)
        connected_components = sorted(connected_components, key=lambda connected_component: len(connected_component), reverse=True)

        # Throw away small connected components.
        connected_components = [connected_component for connected_component in connected_components if len(connected_component) > 10]

        # Compute bounding boxes for the components.
        bounding_boxes = [geometry_utils.get_bounding_box(connected_component) for connected_component in connected_components]

        # Debug: plot bounding boxes
        mask_with_boxes = mask.copy()
        for bounding_box in bounding_boxes:
            for point in geometry_utils.get_line_segment(bounding_box[0], bounding_box[1]):
                mask_with_boxes[point.y][point.x] = 1.0
        utils.show(mask_with_boxes)

        # Compute size of bounding box of a single half note, based on the staff height.
        half_note_height = staff_height / 4.0

        # Split bounding boxes significantly larger than that for a half note into two, either horizontally or diagonally.
        new_boxes = []
        indices_to_delete = []
        for i, bounding_box in enumerate(bounding_boxes):
            bounding_box_height = bounding_box[1].y - bounding_box[0].y
            if bounding_box_height > half_note_height * 1.5:
                indices_to_delete.append(i)
                num_new_boxes = round(bounding_box_height / half_note_height)
                new_box_height = bounding_box_height / num_new_boxes
                new_boxes.extend(
                    (Point(bounding_box[0].y + j * new_box_height, bounding_box[0].x),
                    Point(bounding_box[0].y + (j + 1) * new_box_height, bounding_box[1].x))
                        for j in range(num_new_boxes))
        
        bounding_boxes = [bounding_box for i, bounding_box in enumerate(bounding_boxes) if i not in indices_to_delete]
        bounding_boxes.extend(new_boxes)
        
        # Extract pixels for each bounding box.
        half_notes = []
        for bounding_box in bounding_boxes:
            pixels = [Point(y, x) 
                for y in range(bounding_box[0].y, bounding_box[1].y) 
                for x in range(bounding_box[0].x, bounding_box[1].x)
                if mask[y][x] == 1.0
            ]
            half_notes.append(ScoreElement(ScoreElementType.half_note, pixels))

        # Debug: plot bounding boxes
        mask_with_boxes = mask.copy()
        for half_note in half_notes:
            bounding_box = geometry_utils.get_bounding_box(half_note.pixels)
            for point in geometry_utils.get_line_segment(bounding_box[0], bounding_box[1]):
                mask_with_boxes[point.y][point.x] = 1.0
        utils.show(mask_with_boxes, 'Bounding boxes')

        return half_notes

        ''' 
        # Alternative version to find a first best line
        lines = []
        for skew in range(round(-image.shape[0] * 0.02), round(image.shape[0] * 0.02), 2):
            for start_y in range(max(0, -skew) + image.shape[0] // 4, image.shape[0] // 2):
                point1, point2 = (start_y, 0), (start_y + skew, image.shape[1] - 1)

                # Walk along line throughout the entire image and note the sum of the pixel values in the image along this line.
                line = utils.get_line(point1, point2, mask.shape[0], mask.shape[1])
                line_pixel_sum = sum(mask[y][x] for y, x in line)

                lines.append((line, line_pixel_sum))
            
        lines = sorted(lines, key=lambda line: line[1], reverse=True)
        best_line = lines[0][0]
        '''

       # Debug: plot lines
        '''
        mask_with_kept_notes = mask.copy()
        for half_note in half_notes:
            for (y, x) in line[0]:
                mask_with_lines[y][x] = 1.0
        utils.show(mask_with_lines)
        '''