from typing import Iterable
from aipollo_omr.score_elements import ScoreElement, ScoreElementType
from aipollo_omr.detectors.geometry_utils import Point
import numpy as np
import random
import string
import cv2
import collections
import torch

Tile = collections.namedtuple('Tile', 'data start_y start_x')


def image_to_tiles(image, tile_height=512, tile_width=512):
    tiles = []
    start_ys = list(range(0, image.shape[0] - tile_height + 1, tile_height))
    if image.shape[0] % tile_height != 0:
        start_ys.append(image.shape[0] - tile_height)

    start_xs = list(range(0, image.shape[1] - tile_width + 1, tile_width))
    if image.shape[1] % tile_width != 0:
        start_xs.append(image.shape[1] - tile_width)

    for start_y in start_ys:
        for start_x in start_xs:
            tile_data = np.copy(image[start_y:start_y + tile_height,
                                      start_x:start_x + tile_width])
            tile_data = tile_data.reshape(1, tile_data.shape[0],
                                          tile_data.shape[1])
            tiles.append(Tile(tile_data, start_y, start_x))

    return tiles


def classify(image, model):
    image = cv2.normalize(image,
                          None,
                          alpha=0,
                          beta=1,
                          norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)
    image_tiles = image_to_tiles(image)
    mask = np.zeros(shape=(image.shape[0], image.shape[1]))

    with torch.no_grad():
        for tile in image_tiles:
            tile_mask = model(
                torch.from_numpy(np.reshape(tile.data, (1, *tile.data.shape))))
            mask[tile.start_y:tile.start_y + tile.data.shape[1],
                 tile.start_x:tile.start_x +
                 tile.data.shape[2]] = tile_mask[0, 0, :, :].numpy()

    return mask


def show(mask, window_title=None):
    if not window_title:
        window_title = ''.join(random.choice(string.ascii_uppercase))

    cv2.imshow(window_title, mask)
    cv2.imwrite('foobar.png', mask)
    cv2.waitKey(1)


def draw_score_annotations(score_image, score_elements: Iterable[ScoreElement]):
    score_image = cv2.cvtColor(score_image, cv2.COLOR_GRAY2RGB)

    color_map = {
        ScoreElementType.staff_line: (255, 0, 0),
        ScoreElementType.half_note: (0, 255, 0),
        ScoreElementType.quarter_note: (0, 0, 255),
    }

    def _draw(score_element):
        for pixel in score_element.pixels:
            score_image[pixel.y][pixel.x] = color_map[score_element.type]

        for child in score_element.children:
            _draw(child)

    for score_element in score_elements:
        _draw(score_element)

    show(score_image, window_title='Annotated score')


def draw_line_segments(image, line_segments):
    image_with_lines = image.copy()
    image_with_lines //= 2

    for line_segment in line_segments:
        for point in line_segment:
            image_with_lines[point.y][point.x] = 255

    return image_with_lines