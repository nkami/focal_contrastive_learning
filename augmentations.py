import torch
import torchvision
import random
from PIL import ImageFilter
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_augmentations(selected_augmentations: str) -> list:
    """
    returns a list of transforms used in the official github repo: https://github.com/facebookresearch/moco
    or customs ones.
    :param selected_augmentations: what augmentations to return.
    :return: a list of transforms
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if selected_augmentations == "imagenet_validation":
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    elif selected_augmentations == "moco_v1":
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif selected_augmentations == "moco_v2":
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        raise Exception(f"selected augmentation: {selected_augmentations} is not supported")
    return augmentation


def _main() -> None:
    """
    show random augmented image pairs from the imagenette2 dataset
    :return:
    """
    selected_augmentations = 'moco_v2'
    augmentation = get_augmentations(selected_augmentations)
    ds = torchvision.datasets.ImageFolder('imagenette2/train', TwoCropsTransform(transforms.Compose(augmentation)))
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    for image in dl:
        im1 = image[0][0].reshape((3, 224, 224)).detach().cpu().numpy()
        im2 = image[0][1].reshape((3, 224, 224)).detach().cpu().numpy()
        im1 = np.transpose(im1, (1, 2, 0))
        im2 = np.transpose(im2, (1, 2, 0))
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(im1)
        axarr[1].imshow(im2)
        plt.show()


if __name__ == '__main__':
    _main()
