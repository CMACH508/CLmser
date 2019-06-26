import os
import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import math
from sklearn.manifold import TSNE
import torch.nn.functional as F
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def load_model(module, name, path, args=None, device='cuda:0'):
    """
    prepare classifier and defense model
    :return: defense_model, classifier in eval state
    """
    # build
    model = getattr(module, name)(**args)
    model = model.to(device)
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD).to(x.get_device()) + torch.Tensor(MEAN).to(x.get_device())
    x = x.transpose(1, 3)
    return x


def select_act_func(actfun):
    if actfun == None:
        return lambda x: x
    else:
        return eval('F.' + actfun)


def read_form_list(file, num):
    flist = []
    with open(file, 'r') as f:
        for line in f:
            flist.append(list(line.strip('\n').split(',')))
        return flist


def tsne_(inputs, labels, title):
    y_tsne = TSNE(n_components=2).fit_transform(inputs)
    plot_tsne_embedding(y_tsne, labels, title)
    plt.show()
    return


def plot_tsne_embedding(data, label, title):
    """ plot tsne embedding """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.tab10(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def cal_meancode(code, labels, flag, batch_size):
    """calculate mean code of specific class"""
    indexes = []
    for i in range(batch_size):
        if labels[i] == flag:
            indexes.append(i)
    codes = []
    for j in indexes:
        codes.append(code[j])
    mean_code = np.round(np.mean(codes, axis=0), 2)
    return mean_code


def _save_image(img_dir, image, name):
    file_name = img_dir + '/' + name + '.jpg'
    nrow = int(math.sqrt(image.shape[0]))
    save_image(image.cpu(), file_name, nrow=nrow, padding=2, normalize=True)
    return
