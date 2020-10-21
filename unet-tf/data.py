import pathlib
import numpy as np
import tensorflow as tf
import cv2
import os
import random
from collections import defaultdict

COLAB = False
if COLAB:
    from google.colab.patches import cv2_imshow 


deepscore_path = pathlib.Path('./drive/My Drive/Privates/Coding/aipollo/data/deep_scores_dense_extended') if COLAB else pathlib.Path('C:/Users/simon/Google Drive/Privates/Coding/aipollo/data/deep_scores_dense_extended')

def _get_image_path(image_name):
    return str(deepscore_path / 'images_png') + '/' + image_name

def _get_mask_path(image_name):
    return str(deepscore_path / 'pix_annotations_png') + '/' + image_name


def _extract_staves_as_images(image_name: str):
    full_score_image_path = get_image_path(image_name)
    full_mask_path = get_mask_path(image_name)
    print(full_score_image_path)

    score_image = cv2.imread(full_score_image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)

    try:
        staves_positions = find_max_symmetric_staves_positions(score_image)
    except Exception:
        print(f'Could not extract staves positions {image_name}')
        return []

    staves_images = []
    for start, end in staves_positions:
        stave_image = score_image[start:end, :]
        stave_mask = mask[start:end, :]

        # Resize and embed in white background
        new_dimension = (IMG_WIDTH, (int)(stave_image.shape[0] * (IMG_WIDTH / stave_image.shape[1])))

        stave_image = cv2.resize(stave_image, new_dimension)
        white_image = np.full((IMG_HEIGHT, IMG_WIDTH), 255, np.uint8)
        y_start = (int)(IMG_HEIGHT / 2.0) - (int)(stave_image.shape[0] / 2.0)
        white_image[y_start:y_start + stave_image.shape[0], :stave_image.shape[1]] = stave_image
        stave_image = white_image

        stave_mask = cv2.resize(stave_mask, new_dimension)
        zero_image = np.full((IMG_HEIGHT, IMG_WIDTH), 0, np.uint8)
        zero_image[y_start:y_start + stave_mask.shape[0], :stave_mask.shape[1]] = stave_mask
        stave_mask = zero_image

        #cv2.imshow('stave', stave_image)
        #cv2.imshow('mask', stave_mask)
        #cv2.waitKey(0)
        staves_images.append((stave_image, stave_mask))
    
    return staves_images

def _extract_staves_parts_as_images(image_name):
    full_score_image_path = get_image_path(image_name)
    full_mask_path = get_mask_path(image_name)
    print(full_score_image_path)

    score_image = cv2.imread(full_score_image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)

    try:
        staves_positions = find_symmetric_staves_multiple(score_image, lines_around=3)
    except Exception:
        print(f'Could not extract staves positions {image_name}')
        return []
    
    staves_images = []
    for start, end in staves_positions:
        for start_x in range(0, score_image.shape[1], IMG_WIDTH):
            if start_x + IMG_WIDTH < score_image.shape[1]:
                stave_image = score_image[start:end, start_x:min(start_x + IMG_WIDTH, score_image.shape[1])]
                stave_mask = mask[start:end, start_x:min(start_x + IMG_WIDTH, score_image.shape[1])]
            else:
                stave_image = score_image[start:end, (score_image.shape[1] - IMG_WIDTH):score_image.shape[1]]
                stave_mask = mask[start:end, (score_image.shape[1] - IMG_WIDTH):score_image.shape[1]]

            # Resize and embed in white background
            new_dimension = (IMG_WIDTH, (int)(stave_image.shape[0] * (IMG_WIDTH / stave_image.shape[1])))
            print(stave_image.shape)
            print(new_dimension)

            stave_image = cv2.resize(stave_image, new_dimension)
            white_image = np.full((IMG_HEIGHT, IMG_WIDTH), 255, np.uint8)
            y_start = (int)(IMG_HEIGHT / 2.0) - (int)(stave_image.shape[0] / 2.0)
            white_image[y_start:y_start + stave_image.shape[0], :stave_image.shape[1]] = stave_image
            stave_image = white_image

            stave_mask = cv2.resize(stave_mask, new_dimension)
            zero_image = np.full((IMG_HEIGHT, IMG_WIDTH), 0, np.uint8)
            zero_image[y_start:y_start + stave_mask.shape[0], :stave_mask.shape[1]] = stave_mask
            stave_mask = zero_image

            cv2.imshow('stave', stave_image)
            cv2.imshow('mask', stave_mask)
            #cv2.waitKey(0)
            staves_images.append((stave_image, stave_mask))
    
    return staves_images
    

def convert_images_png():
    pathlib.Path(extracted_staves_path).mkdir(parents=True, exist_ok=True)
    instances = []
    for image_name in os.listdir(str(deepscore_path / 'images_png')):
        if image_name + '_0_mask.png' not in os.listdir(extracted_staves_path):
            staves = extract_staves_parts_as_images(image_name)
            for i in range(len(staves)):
                mask_filename = _extracted_staves_path + '/' + image_name + '_' + str(i) + '_mask.png'
                image_filename = _extracted_staves_path + '/' + image_name + '_' + str(i) + '_image.png'
                print(f'Save to {image_filename}')
                cv2.imwrite(mask_filename, staves[i][1])
                cv2.imwrite(image_filename, staves[i][0])

def get_random_instance(img_height=None, img_width=None):
    image_name = random.choice(os.listdir(str(deepscore_path / 'images_png')))
    print(f'Returning a random image: {image_name}')
    image_path = _get_image_path(image_name)
    mask_path = _get_mask_path(image_name)

    score_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if not(img_height == None) and not(img_height == None):
        score_image = score_image[:img_height, :img_width]
        mask = mask[:img_height, :img_width]

    return score_image, mask


class DataProvider:

    def __init__(self, img_height, img_width, labels_to_use=list(range(0, 158)), one_hot=False):
        self.img_height = img_height
        self.img_width = img_width
        self.one_hot = one_hot

        labels_to_use.append(0)
        labels_to_use = list(set(labels_to_use))
        labels_to_use.sort()
        self.relabeling_dict = defaultdict(int)
        for original_label in range(0, 158):
            if original_label in labels_to_use:
                self.relabeling_dict[original_label] = labels_to_use.index(original_label)


    def yield_data(self):
        extracted_staves_path = str(deepscore_path / 'extracted_staves_') + str(self.img_height) + 'x' + str(self.img_width)
        for image_name in os.listdir(str(deepscore_path / 'images_png')):
            i = 0
            while image_name + '_' + str(i) + '_mask.png' in os.listdir(extracted_staves_path):
                mask_filename = extracted_staves_path + '/' + image_name + '_' + str(i) + '_mask.png'
                image_filename = extracted_staves_path + '/' + image_name + '_' + str(i) + '_image.png'

                score_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

                # Relabel classes
                for index, value in np.ndenumerate(mask):
                    mask[index] = self.relabeling_dict[value]

                # Apply one-hot encoding to the mask
                if self.one_hot:
                    mask = tf.keras.utils.to_categorical(mask, num_classes=len(self.relabeling_dict))
                
                score_image = score_image.reshape(self.img_height, self.img_width, 1)

                print('\n' + f'Providing image {mask_filename}')
                i += 1
                yield score_image, mask


    def get_data(self, stop_when_above=2):
        X = []
        y = []
        for image_name in os.listdir(str(deepscore_path / 'images_png')):
            if len(X) >= stop_when_above:
                break
            i = 0
            while image_name + '_' + str(i) + '_mask.png' in os.listdir(extracted_staves_path):
                mask_filename = extracted_staves_path + '/' + image_name + '_' + str(i) + '_mask.png'
                image_filename = extracted_staves_path + '/' + image_name + '_' + str(i) + '_image.png'

                score_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

                # One-hot convert mask
                mask = tf.keras.utils.to_categorical(mask, num_classes=NUM_CLASSES)

                score_image = score_image.reshape(IMG_HEIGHT, IMG_WIDTH, 1)

                X.append(score_image)
                y.append(mask)

                i += 1

        X = np.asarray(X)
        y = np.asarray(y)
        return X, y
    
    def get_number_classes(self):
        return len(set(self.relabeling_dict.values()))

