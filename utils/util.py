import json
import pandas as pd
import torch
import math
from torchvision.utils import save_image
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def _save_image(img_dir, image, name):
    file_name = img_dir + '/' + name + '.jpg'
    nrow = int(math.sqrt(image.shape[0]))
    save_image(image.cpu(), file_name, nrow=nrow, padding=2, normalize=True)
    return


def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)


def postprocess(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD).cuda() + torch.Tensor(MEAN).cuda()
    x = x.transpose(1, 3)
    return x


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
