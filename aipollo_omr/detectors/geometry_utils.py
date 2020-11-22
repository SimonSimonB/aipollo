import math
import queue
import numpy as np
import cv2
from typing import Iterable


class Point:

    def __init__(self, y, x):
        self.y = int(round(y))
        self.x = int(round(x))

    def __rmul__(self, a):
        return Point(a * self.y, a * self.x)


def get_line_segment(point1, point2):
    if point1.x > point2.x:
        point1, point2 = point2, point1

    if point1.x == point2.x:
        return [Point(y, point1.x) for y in range(point1.y, point2.y + 1)]
    else:
        slope = (point2.y - point1.y) / (point2.x - point1.x)
        return [
            Point(point1.y + slope * (x - point1.x), x)
            for x in range(point1.x, point2.x + 1)
        ]


def get_rotated_bounding_box(pixels: Iterable[Point]):
    raise NotImplementedError


def get_bounding_box(pixels: Iterable[Point]):
    if not pixels:
        return []
    else:
        return (Point(min(point.y for point in pixels),
                      min(point.x for point in pixels)),
                Point(max(point.y for point in pixels),
                      max(point.x for point in pixels)))


def get_convex_hull(pixels: Iterable[Point]):
    hull = cv2.convexHull(np.float32([[pixel.y, pixel.x] for pixel in pixels]))
    hull = hull.reshape(-1, 2).astype(np.int32)
    min_y, min_x = int(min(point[0] for point in hull)), int(
        min(point[1] for point in hull))
    max_y, max_x = int(max(point[0] for point in hull)), int(
        max(point[1] for point in hull))

    filled_hull = []
    black_frame = np.zeros(shape=(math.ceil(max_y) + 1 - math.floor(min_y),
                                  math.ceil(max_x) + 1 -
                                  math.floor(min_x))).astype(np.uint8)
    black_frame = cv2.fillConvexPoly(
        black_frame,
        np.array([[point[1] - min_x, point[0] - min_y] for point in hull]), 255)
    for y in range(black_frame.shape[0]):
        for x in range(black_frame.shape[1]):
            if black_frame[y][x] == 255:
                filled_hull.append(Point(y + min_y, x + min_x))

    return filled_hull


def get_connected_components(arr):

    def _dfs(arr, y, x, visited):
        visited[y][x] = True
        component = [(y, x)]
        on_neighbors = [
            (n_y, n_x)
            for n_x in [x - 1, x + 1]
            for n_y in [y - 1, y + 1]
            if 0 <= n_y < visited.shape[0] and 0 <= n_x < visited.shape[1] and
            not visited[n_y][n_x] and arr[n_y][n_x] == 1
        ]

        for neighbor in on_neighbors:
            component.extend(_dfs(arr, neighbor[0], neighbor[1], visited))

        return component

    discovered = np.zeros_like(arr, dtype=bool)
    components = []

    for start_y in range(arr.shape[0]):
        for start_x in range(arr.shape[1]):
            if arr[start_y][start_x] == 1.0 and not discovered[start_y][start_x]:
                component = []
                q = queue.Queue()
                q.put((start_y, start_x))
                discovered[(start_y, start_x)] = True

                while not q.empty():
                    y, x = q.get()
                    component.append((y, x))
                    neighbors = [(max(0, y - 1), x),
                                 (min(y + 1, discovered.shape[0] - 1), x),
                                 (y, max(0, x - 1)),
                                 (y, min(x + 1, discovered.shape[1] - 1))]

                    for (n_y, n_x) in neighbors:
                        if not discovered[n_y][n_x] and arr[n_y][n_x] == 1:
                            discovered[(n_y, n_x)] = True
                            q.put((n_y, n_x))

                components.append([Point(y, x) for y, x in component])

    return components


def get_line(point1: Point, point2: Point, height, width):
    if point1.x == point2.x:
        return [(y, point1.x) for y in range(height)]

    if point2.x < point1.x:
        point1, point2 = point2, point1

    slope = (point2.y - point1.y) / (point2.x - point1.x)
    intersect = (point1.y - point1.x * slope)

    line = []
    for x in range(width):
        y = round(intersect + slope * x)

        if 0 <= y < height:
            line.append(Point(y, x))

    return line
