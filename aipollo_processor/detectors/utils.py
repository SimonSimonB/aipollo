import queue
import numpy as np
import random
import string
import cv2
import collections
import torch

Tile = collections.namedtuple('Tile', 'data start_y start_x')
def image_to_tiles(image, tile_height=512, tile_width=512):
    #snippet_image = cv2.resize(snippet_image, (snippet_image.shape[1] // downsampling_factor, snippet_image.shape[0] // downsampling_factor))

    tiles = []
    start_ys = list(range(0, image.shape[0] - tile_height + 1, tile_height))
    if image.shape[0] % tile_height != 0:
        start_ys.append(image.shape[0] - tile_height)

    start_xs = list(range(0, image.shape[1] - tile_width + 1, tile_width))
    if image.shape[1] % tile_width != 0:
        start_xs.append(image.shape[1] - tile_width)

    for start_y in start_ys:
        for start_x in start_xs:
            tile_data = np.copy(image[start_y:start_y+tile_height, start_x:start_x+tile_width])
            tile_data = tile_data.reshape(1, tile_data.shape[0], tile_data.shape[1])
            tiles.append(Tile(tile_data, start_y, start_x))

    return tiles


def classify(image, model):
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image_tiles = image_to_tiles(image)
    mask = np.zeros(shape=(image.shape[0], image.shape[1]))

    with torch.no_grad():
        for tile in image_tiles:
            tile_mask = model(torch.from_numpy(np.reshape(tile.data, (1, *tile.data.shape))))
            mask[tile.start_y:tile.start_y+tile.data.shape[1], tile.start_x:tile.start_x+tile.data.shape[2]] = tile_mask[0, 0, :, :].numpy()

    return mask

def show(mask, window_title=None):
    if not window_title:
        window_title = ''.join(random.choice(string.ascii_uppercase))

    cv2.imshow(window_title, mask)
    cv2.waitKey(1)

def get_line(point1, point2, height, width):
    if point1[1] == point2[1]: 
        return [(y, point1[1]) for y in range(height)]

    if point2[1] < point1[1]:
        point1, point2 = point2, point1
    
    slope = (point2[0] - point1[0]) / (point2[1] - point1[1])
    intersect = (point1[0] - point1[1] * slope)

    line = []
    for x in range(width):
        y = round(intersect + slope * x)

        if 0 <= y < height:
            line.append((y, x))
    
    return line

def get_connected_components(arr):

    def _dfs(arr, y, x, visited):
        visited[y][x] = True
        component = [(y, x)]
        on_neighbors = [(n_y, n_x) for n_x in [x - 1, x + 1] for n_y in [y - 1, y + 1] 
                            if 0 <= n_y < visited.shape[0] and 0 <= n_x < visited.shape[1] and not visited[n_y][n_x] and arr[n_y][n_x] == 1]

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
                    neighbors = [(max(0, y - 1), x), (min(y + 1, discovered.shape[0] - 1), x), (y, max(0, x - 1)), (y, min(x + 1, discovered.shape[1] - 1))]

                    for (n_y, n_x) in neighbors:
                        if not discovered[n_y][n_x] and arr[n_y][n_x] == 1:
                            discovered[(n_y, n_x)] = True
                            q.put((n_y, n_x))

                components.append(component)
    
    return components


