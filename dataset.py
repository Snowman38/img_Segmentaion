import os
from glob import glob
import torch
import cv2
import albumentations as A
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np


TRS = [
    A.Compose([A.HorizontalFlip(p=0.3), #provide the worse result
            A.VerticalFlip(p=0.6),
            A.RandomRotate90(p=0.3),
            A.OneOf([
                A.Affine(shear=0.3, p=0.8),
                A.ShiftScaleRotate(shift_limit=0.10, rotate_limit=0, border_mode=cv2.BORDER_REPLICATE, p=0.2)
            ], p=0.3)]),
    A.Compose([A.RandomRotate90(p=1)]),
    A.Compose([A.VerticalFlip(p=1), A.HorizontalFlip(p=1)]),
    
]



class ACDCDataset(torch.utils.data.Dataset):
    """ACDC dataset representation.
    
    This class reads the ACDC images and stores them. It can also apply
    transformations while reading the images if the transform pipeline 
    is provided.
    """

    def __init__(self, paths='', mask_exists=True, transform=None, n_aug=0):
        """Initializes the dataset file.
        Note:
            The path to the dataset directory is expected to contain two
            subdirectories: `image` and `mask` where original images and
            corresponding masks are stored. If the `mask` directory does
            not exist, there is an opportunity to store just the images.
        
        Args:
            paths (str | tuple(str)): The path(-s) to the dataset folder(-s)
            mask_exists (bool): Whether the masks are present
            transform (): Augmentation to apply
        """
        super().__init__()

        # Set the dataset parameters
        self.mask_exists = mask_exists
        self.transform = transform

        # Initialize image and mask lists
        self.img_files = []
        self.mask_files = []

        if isinstance(paths, str):
            # If it's a single path, convert to tuple
            paths = [paths]

        for path in paths:
            # Read the original image files and initialize mask list
            img_files = glob(os.path.join(path, "image", "*.png"))
            mask_files = []

            if mask_exists:
                # If masks exist, read all the mask files
                for img_path in img_files:
                    # Get the mask image file name
                    base = os.path.basename(img_path)
                    mask = os.path.join(path, 'mask', base[:-4] + '_mask.png')

                    # Append the mask file to the list
                    mask_files.append(mask)
            
            # Append image and mask files to the full list
            self.img_files += img_files
            self.mask_files += mask_files
        
        # Generate image and mask values
        self.images = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in self.img_files]
        self.masks = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in self.mask_files]
        # self.images = [Image.open(file) for file in self.img_files]
        # self.masks = [Image.open(file) for file in self.mask_files]
        # Initialize augmentation lists
        img_augs = []
        mask_augs = []
        
        for i in range(n_aug):
            if mask_exists and self.transform is not None:
                for img, msk in zip(self.images, self.masks):
                    # Apply augmentation for images and masks
                    #augmentation = self.transform(image=img, mask=msk)
                    augmentation = TRS[i](image=img, mask=msk)
                    img_augs.append(augmentation["image"])
                    mask_augs.append(augmentation["mask"])
                    #img_augs.append(self.transform(img))
                    #mask_augs.append(self.transform(msk))
            elif self.transform is not None:
                for img in self.images:
                    # Apply augmentation for images
                    augmentation = self.transform(image=img)
                    img_augs.append(augmentation["image"])
        
        # Append augmented images and masks to original ones
        self.images += img_augs
        self.masks += mask_augs


    def __getitem__(self, index):
        """Gets the image(s) at the requested index as float array.
        
        Args:
            index (int): The index of the desired image
        
        Returns:
            (tuple): A pair of image arrays, one representing the
                     original image, the other representing the mask.
                     If there are no masks, a single data array instead
                     of a tuple is returned.
        """
        # Get the image at the specified index
        image = self.images[index]

        if not self.mask_exists:
            # If mask doesn't exist, it's just a single image
            return torch.from_numpy(image).float()

        # Get the massk at the specified index
        mask = self.masks[index]
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()


    def __len__(self):
        """Gets the length of the dataset.
        Return:
            (int): The number of images the dataset contains
        """
        return len(self.images)



def get_mean_std(loader):
    """Calculates the mean and the std of the leaded dataset.
    
    Args:
        loader (DataLoader): The dataset loader
    
    Returns:
        (tuple(float)): A claculated mean and the std of the dataset
    """
    # Initialize the temporary variables
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in loader:
        if len(data.shape) < 4:
            # Reshape properly
            data = data[:, None, ...]
        
        # Sum the running mean and std values for one batch
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    
    # Calculate the mean and std over all batches
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** .5

    return mean, std


def get_transform(augment=True, normalize=True, full=False):
    """Gets the sequence of transformations

    Args:
        augment (bool): Whether augmentation should be applied
        normalize (bool): Whether normalization should be applied
        full (bool): Whether the training is done with validation data

    Returns:
        (A.Compose): A sequence of augmentations to apply
    """
    # Initialize the sequence of compositions
    compose = []

    if augment:
        # Append augmentations to be applied
        compose += [
        
            A.HorizontalFlip(p=0.3), #provide the worse result
            A.VerticalFlip(p=0.6),
            A.RandomRotate90(p=0.3),
            A.OneOf([
                A.Affine(shear=0.3, p=0.8),
                A.ShiftScaleRotate(shift_limit=0.10, rotate_limit=0, border_mode=cv2.BORDER_REPLICATE, p=0.2)
            ], p=0.3)
            
            
        ]
    
    if normalize:
        # Get the precalculated values for mean andstd
        mean = 70.7673 if full else 68.1232
        std = 61.2449 if full else 61.0349

        # Append the augmentation for compose
        compose += [A.Normalize(mean=[mean], std=[std])]
    
    # transform = torchvision.transforms.Compose([
    #     #Rotate by angle degrees
    #     # transforms.RandomRotation(degrees),
    #     #Flip horizontally
    #     transforms.RandomHorizontalFlip(),
    #     #Invert vertically
    #     transforms.RandomVerticalFlip()
    # ])
        
    return A.Compose(compose)


def get_data_loader(
        dataset_path,
        batch_size=16,
        num_workers=0,
        shuffle=True,
        augment=True,
        normalize=True,
        n_aug=0,
        full=False,
        mask_exists=True,
    ):
    """Gets the data loader.

    Args:
        dataset_path (str): The path to the dataset with `image` dir
        batch_size (int): The batch size
        num_workers (int): The number of workers on the loader
        shuffle (bool): Whether to shuffle the loaded entires
        augment (bool): Whether augmentation should be applied
        normalize (bool): Whether normalization should be applied
        n_aug (int): The number of extra augmented datasets to append
        full (bool): Whether the training is done with validation data
        mask_exists (bool): Whether the `mask` subdirectory exists
    
    Returns:
        (DataLoader): The data loader for training/evaluation
    """
    # Get the augmentation to apply
    transform = get_transform(augment, normalize, full)

    # Create the dataset and the data loader
    dataset = ACDCDataset(dataset_path, mask_exists, transform, n_aug)
    loader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

    return loader