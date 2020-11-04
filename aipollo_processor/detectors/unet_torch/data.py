from collections import defaultdict
import random
import tqdm
import pathlib
import numpy as np
import torch
import cv2
import os
from .. import utils

COLAB = False
if COLAB:
    from google.colab.patches import cv2_imshow 


deepscore_path = pathlib.Path('./drive/My Drive/Privates/Coding/aipollo/data/deep_scores_dense_extended') if COLAB else pathlib.Path('C:/Users/simon/Google Drive/Privates/Coding/aipollo/data/deep_scores_dense_extended')
cache_path = pathlib.Path('C:/Users/simon/Coding/ML/aipollo/aipollo_processor/detectors/unet_torch/data/')

def write_to_disk(snippet_height, snippet_width, label_groups, downsampling_factor=3, skip_empty=True):
    snippet_number = 0
    for image_name in tqdm.tqdm(os.listdir(str(deepscore_path / 'images_png'))):
        image_filename = str(deepscore_path / 'images_png') + '/' + image_name
        mask_filename = str(deepscore_path / 'pix_annotations_png') + '/' + image_name
        score_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

        score_image_tiles = utils.image_to_tiles(score_image, tile_height=downsampling_factor * snippet_height, tile_width=downsampling_factor * snippet_width)
        mask_tiles = utils.image_to_tiles(mask, tile_height=downsampling_factor * snippet_height, tile_width=downsampling_factor * snippet_width)

        for score_image_tile, mask_tile in zip(score_image_tiles, mask_tiles):
            snippet_image = score_image_tile.data[0]
            snippet_mask = mask_tile.data[0]

            if snippet_image.mean() == 255:
                print('Skipped a snippet which was all white')
                continue

            snippet_image_path, snippet_mask_path = _get_cache_paths(snippet_height, snippet_width, label_groups, snippet_number)
            if os.path.isfile(snippet_image_path) and os.path.isfile(snippet_mask_path):
                snippet_number += 1
                print('Snippet already exists!')
                continue

            cv2.imshow('Mask snippet before relabeling', cv2.resize(snippet_mask, (snippet_mask.shape[1] // downsampling_factor, snippet_mask.shape[0] // downsampling_factor)))

            if label_groups == [[-1]]:
                staff_lines = [row_index for row_index in range(snippet_mask.shape[0]) if snippet_image[row_index].mean() < 127]  
                for index, _ in np.ndenumerate(snippet_mask):
                    snippet_mask[index] = 1 if snippet_image[index] == 0 and index[0] in staff_lines else 0
            else:
                class_map = defaultdict(lambda: 0)
                for label_group_id, label_group in enumerate(label_groups):
                    for label in label_group:
                        class_map[label] = label_group_id + 1

                for index, value in np.ndenumerate(snippet_mask):
                    snippet_mask[index] = class_map[value]

                snippet_mask = cv2.resize(snippet_mask, (snippet_mask.shape[1] // downsampling_factor, snippet_mask.shape[0] // downsampling_factor))
                snippet_mask = cv2.normalize(snippet_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                snippet_image = cv2.normalize(snippet_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                snippet_image = cv2.resize(snippet_image, (snippet_image.shape[1] // downsampling_factor, snippet_image.shape[0] // downsampling_factor))
                snippet_image = snippet_image.reshape(1, snippet_image.shape[0], snippet_image.shape[1])

                cv2.imshow('Image snippet', snippet_image[0])
                cv2.imshow('Mask snippet', snippet_mask)
                cv2.waitKey(1)

                if snippet_mask.mean() == 0.0:
                    print('Skipped snippet which had no target class pixel')
                    continue

                pathlib.Path(_get_cache_base_path(snippet_height, snippet_width, label_groups)).mkdir(exist_ok=True, parents=True)
                np.savez_compressed(snippet_image_path, snippet_image)
                np.savez_compressed(snippet_mask_path, snippet_mask)

                snippet_number += 1
        
def _get_cache_base_path(snippet_height, snippet_width, label_groups):
    return str(cache_path / f'{snippet_height}x{snippet_width}--{"-".join([str(label) for subgroup in label_groups for label in subgroup])}')

def _get_cache_paths(snippet_height, snippet_width, label_groups, i):
    base_path = _get_cache_base_path(snippet_height, snippet_width, label_groups)
    return (base_path + f'/{i}_image.npz', base_path + f'/{i}_mask.npz') 

class ScoreSnippetsDataset(torch.utils.data.Dataset):

    def __init__(self, snippet_height, snippet_width, label_groups, one_hot=False, transform=None):
        self._snippet_height = snippet_height
        self._snippet_width = snippet_width
        self.one_hot = one_hot
        self._transform = None if not transform else transform
        self._label_groups = label_groups
        self._transform = transform
    
    def __len__(self):
        base_path = _get_cache_base_path(self._snippet_height, self._snippet_width, self._label_groups)
        return len(os.listdir(str(base_path))) // 2

    def __getitem__(self, i):
        score_image_path, mask_path = _get_cache_paths(self._snippet_height, self._snippet_width, self._label_groups, i)
        score_image = np.load(score_image_path)['arr_0']
        mask = np.load(mask_path)['arr_0']

        # Apply transform.
        if self._transform:
            score_image, mask = self._transform((score_image, mask))

        # Apply one-hot encoding to the mask
        if self.one_hot:
            mask = np.eye(len(self._positive_labels), dtype='uint8')[mask]
        
        score_image = score_image.reshape(1, self._snippet_height, self._snippet_width)

        print('\n' + f'Providing image {i}')

        return (score_image, mask)
    
    def _in_cache(self, i):
        score_image_path, mask_path = self._get_cache_paths(i)
        return os.path.isfile(score_image_path) and os.path.isfile(mask_path)

    def write_to_disk(self):
        #cache_folder = cache_path / f'{self._img_height}x{self._img_width}--{"-".join([str(label) for label in self._positive_labels])}'
        #cache_folder.mkdir(parents=True, exist_ok=True)

        for i in range(len(self)):
            score_image_path, mask_path = self._get_cache_paths(i)

            if not self._in_cache(i):
                score_image, mask = self[i]
                np.savez_compressed(score_image_path, score_image)
                np.savez_compressed(mask_path, mask)
                #cv2.imwrite(score_image_path, score_image.reshape(self._img_height, self._img_height, 1))
                #cv2.imwrite(mask_path, mask.reshape(self._img_height, self._img_height, 1))
                print(f'Wrote image #{i}')
