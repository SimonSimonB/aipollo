import os
import torch
import cv2

from aipollo_omr.score_elements import ScoreElement, ScoreElementType
from . import utils
from .unet_torch import models
from . import geometry_utils
from .geometry_utils import Point


class NoteDetector:

    def __init__(self, models_dir):
        model_path = {
            ScoreElementType.half_note: r'half_notes.pt',
            ScoreElementType.quarter_note: r"quarter_notes.pt",
        }

        self._nn = {key: models.UNet() for key in model_path.keys()}
        for note_type in self._nn.keys():
            self._nn[note_type].load_state_dict(
                torch.load(os.path.join(models_dir, model_path[note_type])))
            self._nn[note_type].eval()
        torch.no_grad()

        self._detection_friendly_staff_height = 30

    def detect(self, image, staff_height):
        # Rescale image so that staff height equals detection-friendly staff height.
        resize_factor = self._detection_friendly_staff_height / staff_height
        image = cv2.resize(image, (round(image.shape[1] * resize_factor),
                                   round(image.shape[0] * resize_factor)))

        # Detect notes of each type.
        notes = []
        for note_type, _ in self._nn.items():
            notes_of_type = self.detect_notes_of_type(image, note_type,
                                                      staff_height)

            # Convert detected pixels in resized image to corresponding pixels in original image.
            for note in notes_of_type:
                note_pixels_in_original_image = geometry_utils.get_convex_hull([
                    (1 / resize_factor) * point for point in note.pixels
                ])

                notes.append(
                    ScoreElement(note.type, note_pixels_in_original_image))

            print(f'Found {len(notes_of_type)} of type {note_type}.')

        return notes

    def _split_bounding_boxes(self, bounding_boxes, ideal_bounding_box_height):
        new_boxes = []
        indices_to_delete = []
        for i, bounding_box in enumerate(bounding_boxes):
            bounding_box_height = bounding_box[1].y - bounding_box[0].y
            if bounding_box_height > ideal_bounding_box_height * 1.5:
                indices_to_delete.append(i)
                num_new_boxes = round(bounding_box_height /
                                      ideal_bounding_box_height)
                new_box_height = bounding_box_height / num_new_boxes
                for j in range(num_new_boxes):
                    jth_new_box_top_left = Point(
                        bounding_box[0].y + j * new_box_height,
                        bounding_box[0].x)
                    jth_new_box_bottom_right = Point(
                        bounding_box[0].y + (j + 1) * new_box_height - 1,
                        bounding_box[1].x)
                    new_boxes.append(
                        (jth_new_box_top_left, jth_new_box_bottom_right))

        bounding_boxes = [
            bounding_box for i, bounding_box in enumerate(bounding_boxes)
            if i not in indices_to_delete
        ]
        bounding_boxes.extend(new_boxes)

        return bounding_boxes

    def detect_notes_of_type(self, image, note_type, staff_height):
        utils.show(image)
        mask = utils.classify(image, self._nn[note_type])

        # Threshold the pixel-wise classification
        threshold = 0.1
        mask[mask > threshold] = 1.0
        mask[mask <= threshold] = 0.0
        utils.show(mask)

        connected_components = geometry_utils.get_connected_components(mask)
        connected_components = sorted(
            connected_components,
            key=lambda connected_component: len(connected_component),
            reverse=True)

        # Throw away small connected components.
        connected_components = [
            connected_component for connected_component in connected_components
            if len(connected_component) > 10
        ]

        # Compute bounding boxes for the components.
        bounding_boxes = [
            geometry_utils.get_bounding_box(connected_component)
            for connected_component in connected_components
        ]

        # Debug: plot bounding boxes
        mask_with_boxes = mask.copy()
        for bounding_box in bounding_boxes:
            for point in geometry_utils.get_line_segment(
                    bounding_box[0], bounding_box[1]):
                mask_with_boxes[point.y][point.x] = 1.0
        utils.show(mask_with_boxes)

        # Split bounding boxes significantly higher than a single note: they probably encompass several notes.
        notehead_height = staff_height / 4.0
        bounding_boxes = self._split_bounding_boxes(
            bounding_boxes, ideal_bounding_box_height=notehead_height)

        # Extract pixels for each bounding box.
        notes = []
        for bounding_box in bounding_boxes:
            pixels = [
                Point(y, x)
                for y in range(bounding_box[0].y, bounding_box[1].y)
                for x in range(bounding_box[0].x, bounding_box[1].x)
                if mask[y][x] == 1.0
            ]
            notes.append(ScoreElement(note_type, pixels))

        # Debug: plot bounding boxes
        mask_with_boxes = mask.copy()
        for note in notes:
            bounding_box = geometry_utils.get_bounding_box(note.pixels)
            for point in geometry_utils.get_line_segment(
                    bounding_box[0], bounding_box[1]):
                mask_with_boxes[point.y][point.x] = 1.0
        utils.show(mask_with_boxes, 'Bounding boxes')

        return notes