import os
import pickle

import anytree
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class GLD160(Dataset):
    data_name = 'GLD'

    def __init__(self, images, targets, transform=T.RandomCrop(92)):
        self.transform = transform
        self.img, self.target = images, targets
        self.classes_counts = 2028

    def __getitem__(self, index):
        img = self.img[index]
        target = self.target[index]
        inp = {'img': img, 'label': target}
        if self.transform is not None:
            inp['img'] = self.transform(inp['img'])
        return inp

    def __len__(self):
        return len(self.img)