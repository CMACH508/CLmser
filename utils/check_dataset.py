from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import shutil

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--path', type=str, default=None, required=True,
                    help='Batch size to use')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths, len(img_paths)


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)


def make_dataset(path_files):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files)
    else:
        paths, size = make_dataset_dir(path_files)

    return paths, size


def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)

    return img_paths, len(img_paths)


def check_repeat():
    tab = {}
    files, file_length = make_dataset('new_test.txt')
    print(file_length)
    for i in tqdm(range(len(files))):

        if str(files[i]) in tab.keys():
            tab[str(files[i])] += 1
        else:
            tab[str(files[i])] = 0

    print(max(tab.values()))


def check_bad_image(path):
    files, file_length = make_dataset(path)
    count = 0
    for i in tqdm(range(len(files))):
        x = np.array(imread(files[i]).astype(np.float32))
        if x.shape != (256, 256, 3):
            count += 1
            # name = os.path.split(files[i])[1]
            # shutil.copyfile(files[i], os.path.join('bad_images', name))
    print(count)


if __name__ == '__main__':
    args = parser.parse_args()
    check_bad_image(args.path)
