import random
import unet_torch.models
import utils
import torch
import cv2

class StaffDetector:

    def __init__(self):
        self._nn = unet_torch.models.UNet()
        self._nn.load_state_dict(torch.load('aipollo_processor/detectors/unet_torch/logs/[-1]--2020-10-29-18.37.07/4500.pt'))
        self._nn.eval()
        torch.no_grad()

        self._detection_friendly_staff_height = 30


    def detect(self, image):
        # Find staffs in image that is not optimally scaled yet since we do not know staff height. Try to detect staff height in that image.
        resize_to_height = 1024
        image = cv2.resize(image, (round(image.shape[1] * (resize_to_height / image.shape[0])), resize_to_height))
        staffs = self._detect(image)
        if not staffs:
            return []

        staff_height = sum(staff[4][0][0][0] - staff[0][0][0][0] for staff in staffs) / len(staffs)

        # Rescale image so that staff height equals detection-friendly staff height.
        resize_factor = self._detection_friendly_staff_height / staff_height
        image = cv2.resize(image, (round(image.shape[1] * resize_factor), round(image.shape[0] * resize_factor)))

        # Detect staffs again.
        staffs = self._detect(image)
        print(f'Found {len(staffs)} staffs.')

        return staffs


    def _detect(self, image):
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

        # Find one really good line by fitting lines through pixels in the same connected component in the mask.
        # TODO Can probably speed that up by much by just randomly shifting lines across the image. 
        connected_components = utils.get_connected_components(mask)
        connected_components = sorted(connected_components, key=lambda connected_component: len(connected_component), reverse=True)
        lines = []
        for connected_component in connected_components[:30]:
            line_candidates = []
            for _ in range(50):
                # Pick two random points.
                point1, point2 = random.sample(connected_component, 2)

                # Walk along line throughout the entire image and note the sum of the pixel values in the image along this line.
                line = utils.get_line(point1, point2, mask.shape[0], mask.shape[1])
                line_pixel_sum = sum(mask[y][x] for y, x in line)

                line_candidates.append((line, line_pixel_sum))
            
            best_line = max(line_candidates, key=lambda line_and_line_pixel_sum: line_and_line_pixel_sum[1])
            lines.append(best_line)

        lines = sorted(lines, key=lambda line: line[1], reverse=True)
        best_line = lines[0][0]
        
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


        # Find other lines by shifting the one line we found.
        line_candidates = []
        for shift in range(-best_line[0][0], mask.shape[0] - best_line[0][0]):
            line_candidate = utils.get_line((best_line[0][0] + shift, best_line[0][1]), (best_line[-1][0] + shift, best_line[-1][1]), mask.shape[0], mask.shape[1])
            line_candidate_pixel_sum = sum(mask[y][x] for y, x in line_candidate)
            line_candidates.append([line_candidate, line_candidate_pixel_sum])

        # Throw away all but the 100 lines with the highest fit.
        line_candidates = sorted(line_candidates, key=lambda line: line[1], reverse=True)[:round(0.1 * image.shape[0])]

        # Merge nearby lines. TODO Rewrite this so that line is at center of previous lines.
        line_candidates = sorted(line_candidates, key=lambda line: line[0][0]) 
        i = 0
        while i < len(line_candidates) - 1:
            if line_candidates[i + 1][0][0][0] - line_candidates[i][0][0][0] < 3:
                line_candidates[i][1] += line_candidates[i + 1][1]
                del line_candidates[i+1]
            else:
                i += 1

        # Debug: plot lines
        mask_with_lines = mask.copy()
        for line in line_candidates:
            for (y, x) in line[0]:
                mask_with_lines[y][x] = 1.0
        utils.show(mask_with_lines)

        # Determine the distance between staff lines in this image.
        staff_line_distances = sorted(line_candidates[i + 1][0][0][0] - line_candidates[i][0][0][0] for i in range(len(line_candidates) - 1))
        max_staff_line_distance = staff_line_distances[len(staff_line_distances) // 2] * 1.3

        # Categorize into staffs by assigning staff lines that have a sufficiently small distance into the same staff
        current_component = [line_candidates[0]]
        staffs = [current_component]
        for i in range(1, len(line_candidates)):
            if line_candidates[i][0][0][0] - line_candidates[i - 1][0][0][0] > max_staff_line_distance:
                current_component = []
                staffs.append(current_component)
            
            current_component.append(line_candidates[i])
        
        # Throw staff line groups that have just one member.
        good_staffs = [staff for staff in staffs if len(staff) == 5]

         # Debug: plot lines in mask
        mask_with_lines = mask.copy()
        for line in [line for staff in staffs for line in staff]:
            for (y, x) in line[0]:
                mask_with_lines[y][x] = 1.0
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

        return good_staffs
