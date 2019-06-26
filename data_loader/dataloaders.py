from base import BaseDataLoader
from torchvision import transforms
from data_loader.datasets import Places2, CelebA


class Places2DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, split, img_size, data_dir, batch_size, shuffle, num_workers):
        if split == 'train':
            self.img_transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=img_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])
                 ])

        else:
            self.img_transform = transforms.Compose(
                [transforms.Resize(size=img_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])
                 ])

        dataset = Places2(split, data_dir, self.img_transform)

        super(Places2DataLoader, self).__init__(split, dataset, batch_size, shuffle, num_workers)


class CelebADataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, split, img_size, data_dir, batch_size, shuffle, num_workers):
        self.img_transform = transforms.Compose(
            [transforms.Resize(size=img_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
             ])
        dataset = CelebA(split, data_dir, self.img_transform)

        super(CelebADataLoader, self).__init__(split, dataset, batch_size, shuffle, num_workers)
        super(CelebADataLoader, self).__init__(split, dataset, batch_size, shuffle, num_workers)
