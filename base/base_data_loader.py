from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, split, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):

        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)

        if split == 'train':
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': self.shuffle,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }
            super(BaseDataLoader, self).__init__(**self.init_kwargs)
        else:
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': False,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }
            super(BaseDataLoader, self).__init__(**self.init_kwargs)
