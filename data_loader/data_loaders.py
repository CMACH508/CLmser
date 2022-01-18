import torchvision.transforms as transforms
from .data_set import BaseDataSet
from torch.utils.data import DataLoader


class CelebAHQDataLoader(DataLoader):
    def __init__(self, split, img_dir, img_list, mask_dir, batch_size,
                 img_size, shuffle, num_workers):

        if split == 'train':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            assert shuffle is False
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        self.dataset = BaseDataSet(split, img_list, img_dir, mask_dir, transform)
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        super(CelebAHQDataLoader, self).__init__(**self.init_kwargs)


class Places2DataLoader(DataLoader):
    def __init__(self, split, img_dir, img_list, mask_dir, batch_size,
                 img_size, shuffle, num_workers):
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

            ])
        else:
            assert shuffle is False
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        self.dataset = BaseDataSet(split, img_list, img_dir, mask_dir,  transform)
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }

        super(Places2DataLoader, self).__init__(**self.init_kwargs)

