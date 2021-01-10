from aipollo_omr.detectors.geometry_utils import Point
from aipollo_omr.score_elements import ScoreElement, ScoreElementType
from aipollo_omr.detectors import geometry_utils
import random
from aipollo_omr.detectors.unet_torch import models
from . import utils
import torch
import cv2
import os.path


class StaffDetector:

    def __init__(self, models_dir):
        self._nn = models.UNet()
        model_path = str(os.path.join(models_dir, r'staffs.pt'))
        self._nn.load_state_dict(torch.load(model_path))
        self._nn.eval()
        torch.no_grad()

        self._detection_friendly_staff_height = 30

    def detect(self, image, rescale_to_improve_detection=False):

        if rescale_to_improve_detection:
            staff_height = self._estimate_staff_height(image)
            resize_factor = self._detection_friendly_staff_height / staff_height
        else:
            resize_to_height = 1024
            resize_factor = resize_to_height / image.shape[0]

        resized_image = cv2.resize(image,
                                   (round(image.shape[1] * resize_factor),
                                    round(image.shape[0] * resize_factor)))
        line_groups = self._detect_line_groups(resized_image)
        original_image_line_groups = self._resize_line_groups(
            line_groups, 1 / resize_factor, image.shape[0], image.shape[1])

        staffs = self._convert_line_groups_to_staffs(original_image_line_groups)
        staff_height = self._compute_average_staff_height(staffs)
        print(f'Found {len(staffs)} staffs.')

        return staffs, staff_height

    def _detect_line_groups(self, image):
        utils.show(image)
        mask = utils.classify(image, self._nn)
        self._threshold_mask(mask, 0.1)
        best_line = self._detect_one_good_staff_line(mask)
        lines_and_scores = self._find_best_parallel_lines(
            best_line, mask, num_lines=round(0.1 * image.shape[0]))
        lines_and_scores = self._merge_nearby_lines(lines_and_scores)

        # Debug: plot lines
        mask_with_lines = mask.copy()
        for line in [line_and_score[0] for line_and_score in lines_and_scores]:
            for point in line:
                mask_with_lines[point.y][point.x] = 1.0
        utils.show(mask_with_lines)

        staffs = self._sort_into_staffs(lines_and_scores)

        # Debug: plot lines in mask
        mask_with_lines = mask.copy()
        for staff in staffs:
            for line in staff:
                for point in line:
                    mask_with_lines[point.y][point.x] = 1.0
        utils.show(mask_with_lines, 'Kept staff')

        #TODO Either keep staffs with > 1 members and fill out those that have < 5; or shift one of the 5er staffs you found around to find other staffs.
        '''
        average_staff_height = sum(staff[4][0][0][0] - staff[0][0][0][0] for staff in good_staffs) / len(good_staffs)
        average_staff_line_distance = sum(staff[i][0][0][0] - staff[i - 1][0][0][0] for staff in good_staffs for i in range(1,5)) / (5 * len(good_staffs))

        for staff in staffs:
            if len(staff) == 5:
                continue

            # Fill in missing lines in middle
            for i in range(len(staff) - 1):
                staff_line_distance = staff[i + 1][0][0][0] - staff[i][0][0][0]
                if (staff_line_distance / average_staff_line_distance) > 1.5:
                    staff_lines_to_add = round(staff_line_distance / average_staff_line_distance)
                    space_between_new_lines = staff_line_distance / staff_lines_to_add
                    for j in range(staff_lines_to_add):
                        new_staff_line = [(y + round(space_between_new_lines * (j + 1)), x) for (y, x) in staff[i]]
                        staff.insert(i + j + 1, new_staff_line)
            
            # Expand above and below
            while len(staff) < 5:
                candidates = [
                    [(max(0, y - average_staff_line_distance), x) for (y, x) in staff[0]],
                    [(min(image.height, y + average_staff_line_distance), x) for (y, x) in staff[0]] 
                ]

                candidates_pixel_sum = [sum(mask[y][x] for y, x in line_candidate) for line_candidate in candidates]
                new_staff_line = candidates[candidates_pixel_sum.find(max(candidates_pixel_sum))]
                staff.insert(0 if new_staff_line[0][0][0] < staff[0][0][0][0] else -1, new_staff_line)
            '''

        return staffs

    def _detect_one_good_staff_line(self, mask):

        mask_uint = mask.astype('uint8')
        mask_uint *= 255
        fast_line_detector = cv2.ximgproc.createFastLineDetector(
            10, 1.414, 50, 50, 3, True)
        lines = fast_line_detector.detect(mask_uint)
        line_segments = [
            geometry_utils.get_line_segment(
                geometry_utils.Point(y=line[0][1], x=line[0][0]),
                geometry_utils.Point(y=line[0][3], x=line[0][2]))
            for line in lines
        ]

        # Find one really good line by fitting lines through pixels in the same connected component in the mask.
        line_segments.sort(key=lambda line_segment: len(line_segment),
                           reverse=True)
        lines = []
        for connected_component in line_segments[:10]:
            lines_and_scores = []
            for _ in range(5):
                # Pick two random points.
                point1, point2 = random.sample(connected_component, 2)

                # Walk along line throughout the entire image and note the sum of the pixel values in the image along this line.
                line = geometry_utils.get_line(point1, point2, mask.shape[0],
                                               mask.shape[1])
                line_pixel_sum = sum(mask[p.y][p.x] for p in line)

                lines_and_scores.append((line, line_pixel_sum))

            best_line = max(
                lines_and_scores,
                key=lambda line_and_line_pixel_sum: line_and_line_pixel_sum[1])
            lines.append(best_line)

        lines = sorted(lines, key=lambda line: line[1], reverse=True)
        best_line = lines[0][0]

        return best_line

    def _find_best_parallel_lines(self, best_line, mask, num_lines):
        lines_and_scores = []
        for shift in range(-best_line[0].y, mask.shape[0] - best_line[0].y):
            line_candidate = geometry_utils.get_line(
                geometry_utils.Point(best_line[0].y + shift, best_line[0].x),
                geometry_utils.Point(best_line[-1].y + shift, best_line[-1].x),
                mask.shape[0], mask.shape[1])
            line_candidate_pixel_sum = sum(
                mask[p.y][p.x] for p in line_candidate)
            lines_and_scores.append([line_candidate, line_candidate_pixel_sum])

        lines_and_scores = sorted(lines_and_scores,
                                  key=lambda line: line[1],
                                  reverse=True)[:num_lines]

        return lines_and_scores

    def _merge_nearby_lines(self, lines_and_scores):
        # TODO Rewrite this so that the merged line is at the center of previous lines.
        lines_and_scores = sorted(lines_and_scores,
                                  key=lambda line: line[0][0].y)
        current_line_group = []
        line_groups = []
        i = 0
        while i < len(lines_and_scores) - 1:
            current_line_group.append(lines_and_scores[i])
            if lines_and_scores[i +
                                1][0][0].y - lines_and_scores[i][0][0].y > 3:
                line_groups.append(current_line_group)
                current_line_group = []

            i += 1
        else:
            current_line_group.append(lines_and_scores[i])
            line_groups.append(current_line_group)

        lines_and_scores = []
        for line_group in line_groups:
            # Compute the averaged line.
            start_x = max(
                line_and_score[0][0].x for line_and_score in line_group)
            end_x = min(
                line_and_score[0][-1].x for line_and_score in line_group)
            line = []
            for i in range(end_x - start_x + 1):
                line.append(
                    Point(
                        sum(line_and_score[0][i].y
                            for line_and_score in line_group) / len(line_group),
                        start_x + i))

            score = sum(line_and_score[1] for line_and_score in line_group)
            lines_and_scores.append((line, score))

        return lines_and_scores

    def _sort_into_staffs(self, lines_and_scores):
        # Determine the distance between staff lines in this image.
        staff_line_distances = sorted(lines_and_scores[i + 1][0][0].y -
                                      lines_and_scores[i][0][0].y
                                      for i in range(len(lines_and_scores) - 1))
        max_staff_line_distance = staff_line_distances[len(staff_line_distances)
                                                       // 2] * 1.3

        # Categorize into staffs by assigning staff lines that have a sufficiently small distance into the same staff
        current_staff = [lines_and_scores[0][0]]
        staffs = [current_staff]
        for i in range(1, len(lines_and_scores)):
            if lines_and_scores[i][0][0].y - lines_and_scores[
                    i - 1][0][0].y > max_staff_line_distance:
                current_staff = []
                staffs.append(current_staff)

            current_staff.append(lines_and_scores[i][0])

        staffs = [staff for staff in staffs if len(staff) == 5]

        return staffs

    def _estimate_staff_height(self, image):
        # Detect staff height in the provided image.
        resize_to_height = 1024
        resize_factor = resize_to_height / image.shape[0]
        image = cv2.resize(image, (round(image.shape[1] * resize_factor),
                                   round(image.shape[0] * resize_factor)))
        staffs = self._detect_line_groups(image)

        if len(staffs) == 0:
            raise ValueError('No staffs found in the given image.')

        return self._compute_average_staff_height(staffs)

    def _compute_average_staff_height(self, staffs):
        staff_heights = []
        for staff in staffs:
            staff_height = staff.children[4].pixels[0].y - staff.children[
                0].pixels[0].y
            staff_heights.append(staff_height)

        return sum(staff_heights) / len(staff_heights)

    def _convert_line_groups_to_staffs(self, line_groups):
        staffs = [[
            ScoreElement(ScoreElementType.staff_line, staff_line)
            for staff_line in line_group
        ]
                  for line_group in line_groups]

        staffs = [
            ScoreElement(ScoreElementType.staff, pixels=[], children=staff)
            for staff in staffs
        ]

        return staffs

    def _resize_line_groups(self, line_groups, resize_factor, maxY, maxX):

        return [[
            geometry_utils.get_line(resize_factor * line[0],
                                    resize_factor * line[-1], maxY, maxX)
            for line in line_group
        ]
                for line_group in line_groups]

    def _threshold_mask(self, mask, threshold=0.1):
        mask[mask > threshold] = 1.0
        mask[mask <= threshold] = 0.0
        utils.show(mask)

        return None
