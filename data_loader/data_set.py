import torch
import math
import random
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from data_loader.image_folder import make_dataset
import numpy as np
import os


class BaseDataSet(torch.utils.data.Dataset):
    def __init__(self, split, img_list, img_dir, mask_file,
                 transform=None):
        super(BaseDataSet, self).__init__()
        self.img_paths, self.img_size = make_dataset(img_list)
        self.img_dir = img_dir
        self.mask_paths, self.mask_size = make_dataset(mask_file)
        self.mask_paths = self.mask_paths * (max(1, math.ceil(self.img_size / self.mask_size)))
        self.transform = transform
        self.split = split

    def __getitem__(self, index):
        img, img_path = self.load_img(index)
        mask = self.load_mask(img, index)
        return img, mask, img_path

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = os.path.join(self.img_dir, self.img_paths[index])
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path

    def load_mask(self, img, index):
        if self.split == 'train':
            mask_index = random.randint(0, self.mask_size - 1)
            mask_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=img.size(2),
                                             interpolation=Image.NEAREST),
                transforms.ToTensor()])
        else:
            mask_index = index
            mask_transform = transforms.Compose([
                transforms.Resize(size=img.size(2),
                                  interpolation=Image.NEAREST),
                transforms.ToTensor()])
        mask_pil = Image.open(self.mask_paths[mask_index]).convert('1')
        mask_pil = Image.fromarray(1. - np.array(mask_pil))

        # size = mask_pil.size[0]
        # if size > mask_pil.size[1]:
        #     size = mask_pil.size[1]

        mask = (mask_transform(mask_pil) == 0).float()
        mask_pil.close()
        return mask

    def __len__(self):
        return self.img_size
