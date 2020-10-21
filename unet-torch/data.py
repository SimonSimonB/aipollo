import pathlib
import numpy as np
import torch
import cv2
import os

COLAB = False
if COLAB:
    from google.colab.patches import cv2_imshow 


deepscore_path = pathlib.Path('./drive/My Drive/Privates/Coding/aipollo/data/deep_scores_dense_extended') if COLAB else pathlib.Path('C:/Users/simon/Google Drive/Privates/Coding/aipollo/data/deep_scores_dense_extended')
cache_path = pathlib.Path('C:/Users/simon/Coding/ML/aipollo/unet-torch/data/')

def _get_image_path(image_name):
    return str(deepscore_path / 'images_png') + '/' + image_name

def _get_mask_path(image_name):
    return str(deepscore_path / 'pix_annotations_png') + '/' + image_name
   


class ScoreSnippetsDataset(torch.utils.data.Dataset):

    def __init__(self, img_height, img_width, positive_labels, one_hot=False, transform=None, from_cache=False, custom_size=None):
        self._img_height = img_height
        self._img_width = img_width
        self.one_hot = one_hot
        self._transform = None if not transform else transform
        self._positive_labels = positive_labels
        self._from_cache = from_cache
        self._custom_size = custom_size
        self._transform = transform
    
    def __len__(self):
        return self._custom_size if self._custom_size else len(os.listdir(str(deepscore_path / 'images_png'))) 

    def __getitem__(self, i):
        if self._from_cache and self._in_cache(i):
            score_image_path, mask_path = self._get_cache_paths(i)
            score_image = np.load(score_image_path)['arr_0']
            mask = np.load(mask_path)['arr_0']
        else:
            image_name = os.listdir(str(deepscore_path / 'images_png'))[i]

            image_filename = str(deepscore_path / 'images_png') + '/' + image_name
            mask_filename = str(deepscore_path / 'pix_annotations_png') + '/' + image_name

            score_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            score_image = score_image[100:100+self._img_height, 100:100+self._img_width]
            score_image = cv2.normalize(score_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            mask = mask[100:100+self._img_height, 100:100+self._img_width]
            # Relabel classes
            for index, value in np.ndenumerate(mask):
                #if mask[index] in self._positive_labels:
                #    print(f'Hit {mask[index]}')
                #if mask[index] != 0:
                #    print(mask[index])
                mask[index] = 1 if value in self._positive_labels else 0

            mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            #cv2.imshow('Score', score_image)
            #cv2.waitKey(1)
            #cv2.imshow('All labels', mask)
            #cv2.waitKey(1)

        # Apply transform.
        if self._transform:
            score_image, mask = self._transform((score_image, mask))

        # Apply one-hot encoding to the mask
        if self.one_hot:
            mask = np.eye(len(self._positive_labels), dtype='uint8')[mask]
        
        score_image = score_image.reshape(1, self._img_height, self._img_width)

        print('\n' + f'Providing image {i}')

        return (score_image, mask)
    
    def _get_cache_paths(self, i):
        return (str(cache_path / f'{self._img_height}x{self._img_width}--{"-".join([str(label) for label in self._positive_labels])}' / f'{i}_image.npz'),
            str(cache_path / f'{self._img_height}x{self._img_width}--{"-".join([str(label) for label in self._positive_labels])}' / f'{i}_mask.npz'))

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
