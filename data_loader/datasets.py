import torch
from PIL import Image
from glob import glob


class CelebA(torch.utils.data.Dataset):
    def __init__(self, split, img_root, img_transform):
        super(CelebA, self).__init__()
        self.img_transform = img_transform

        self.paths = glob('{:s}/{:s}/*.jpg'.format(img_root, split))

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))
        return gt_img

    def __len__(self):
        return len(self.paths)


class Places2(torch.utils.data.Dataset):
    def __init__(self, split, img_root, img_transform):
        super(Places2, self).__init__()
        self.img_transform = img_transform

        if split == 'test':
            self.paths = glob('{:s}/test/*.jpg'.format(img_root))

        else:
            self.paths = glob('{:s}/{:s}//**/*.jpg'.format(img_root, split))

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))
        return gt_img

    def __len__(self):
        return len(self.paths)
