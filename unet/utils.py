import pickle
import sklearn
import numpy as np
import random
import cv2

COLOR_MAP_PATH = 'C:/Users/simon/Coding/ML/aipollo/color_map.p'

def build_color_map(num_classes):
    color_map = {}
    for i in range(num_classes):
        color_map[i] = [random.randint(0, 128), random.randint(0, 128), random.randint(0, 128)]

    with open(COLOR_MAP_PATH, 'wb+') as outfile:
        pickle.dump(color_map, outfile)
    
def show_mask(mask, window_name, block=False):
    color_map = {}
    with open(COLOR_MAP_PATH, 'rb') as color_map_file:
        color_map = pickle.load(color_map_file)

    # Build colored predictions
    mask_colored = np.zeros(shape=(mask.shape[0], mask.shape[1], 3), dtype=np.int8)
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            mask_colored[row, column] = color_map[mask[row, column]]
    
    cv2.imshow(window_name, mask_colored)
    cv2.waitKey(0 if block else 1)


def show_prediction(model, x: np.ndarray, block=False):
    x = x.reshape(1, x.shape[0], x.shape[1], 1)
    y_hat = np.asarray(model.predict(x))
    y_hat = np.argmax(y_hat, axis=3)
    y_hat = y_hat[0]
    show_mask(y_hat, 'prediction', block)
    
"""
    Args:
        model: the model to benchmark
        instances: list of pairs of 2D arrays
"""
def benchmark_model(model, instances):
    y_hats = []
    ys = []

    for x, y in instances:
        assert x.ndim == 2 and y.ndim == 2

        x = x.reshape(1, x.shape[0], x.shape[1], 1)
        y_hat = np.asarray(model.predict(x))
        y_hat = np.where(y_hat > 0.5, 1, 0) if (y.ndim == 1 or y_hat.shape[1] == 1)  else np.argmax(y_hat, axis=3)
        y_hat_flattened = y_hat.flatten()

        y_flattened = y.flatten()

        y_hats.extend(y_hat_flattened)
        ys.extend(y_flattened)

    report = sklearn.metrics.classification_report(ys, y_hats, output_dict=True)

    f1_scores = []
    for label in filter(lambda x: x.isdigit(), report.keys()):
        f1_scores.append((label, report[str(label)]['f1-score']))
    
    print('\n')
    print(sorted(f1_scores, key = lambda pair: pair[1], reverse=True))
    print('\n')

    return