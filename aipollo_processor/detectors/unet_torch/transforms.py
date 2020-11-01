import numpy as np
import torchvision
import torch

class RandomResize:
    def __call__(self, sample):
        image, target = sample
        image_height, image_width = image.shape[1], image.shape[2]
        aspect_ratio = max(image_width, image_height) / min(image_width, image_height)

        image *= 255
        target *= 255
        image = image.astype(np.uint8)
        image = image.reshape((image_height, image_width))
        target = target.astype(np.uint8)
        image, target = torchvision.transforms.functional.to_pil_image(image, mode='L'), torchvision.transforms.functional.to_pil_image(target, mode='L')

        rotation = torchvision.transforms.RandomAffine(30, translate=None, scale=(0.3, 1), shear=None, resample=False)
        angle, translations, scale, shear = rotation.get_params(rotation.degrees, rotation.translate, rotation.scale, rotation.shear, (image_height, image_width))
        image = torchvision.transforms.functional.affine(image, angle, translations, scale, shear, fillcolor=255)
        target = torchvision.transforms.functional.affine(target, angle, translations, scale, shear, fillcolor=0)

        random_crop = torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.9, 0.9), ratio=(0.5*aspect_ratio, 2*aspect_ratio))
        i, j, h, w = random_crop.get_params(image, random_crop.scale, random_crop.ratio)
        image = torchvision.transforms.functional.resized_crop(image, i, j, h, w, random_crop.size, random_crop.interpolation)
        target = torchvision.transforms.functional.resized_crop(target, i, j, h, w, random_crop.size, random_crop.interpolation)

        image = torchvision.transforms.ColorJitter(contrast=0.5, saturation=0.5)(image)

        image = torchvision.transforms.ToTensor()(image)
        target = torchvision.transforms.ToTensor()(target)
        target = target.reshape((image_height, image_width))
        image = image.numpy()
        target = target.numpy()

        return (image, target)

class Noise:
    def __call__(self, sample):
        image, target = sample

        mean = 0
        stddev = np.random.uniform(0, 0.4)
        image += np.random.normal(mean, stddev, image.shape)

        return (image, target)


class Normalize:
    def __call__(self, sample):
        image, target = sample
        image += -min(0, image.min())
        image *= 1.0 / image.max()

        return (image, target)