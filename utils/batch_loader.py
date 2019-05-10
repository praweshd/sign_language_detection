from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, datasets
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import os
import numpy as np 

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

class dataset_pipeline(Dataset):

    def __init__(self, csv_file, root_dir, transform=transforms.Compose([ToTensor()])):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.labels_frame.iloc[idx, 0])
        image = io.imread(img_name)
        labels = np.asarray([self.labels_frame.iloc[idx, 1]])

        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample