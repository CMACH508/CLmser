import shutil
import random
from pathlib import Path
import os
import glob

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--path', type=str, default=None, required=True,
                    help='Batch size to use')
parser.add_argument('--new_list', type=str, default=None, required=True,
                    help='Batch size to use')
parser.add_argument('--number', type=int, default=0,
                    help='number')
parser.add_argument('--portion', type=float, default=0,
                    help='portion')
parser.add_argument('--new_dir', type=str, default=None,
                    help='Batch size to use')
parser.add_argument('--has_subdir', type=bool, default=False,
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

    return img_paths


def make_dataset(path_files):
    if path_files.find('.txt') != -1:
        paths = make_dataset_txt(path_files)
    else:
        paths = glob_name_dir(path_files)

    return paths


def glob_name_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        sub_dir = root.split('/')[-1]
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_paths.append(os.path.join(sub_dir, fname))
    return img_paths


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

    return img_paths


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def split_by_all(path, new_list, number=0, portion=0, new_dir=None, has_subdir=False):
    files = make_dataset(path)
    if portion != 0:
        numbers = int(len(files) * portion)
    else:
        assert number != 0
        numbers = number
    samples = random.sample(range(len(files)), numbers)

    if new_dir is not None:
        for i in samples:
            seps = files[i].split('/')
            name = seps[-1]
            if has_subdir:
                sub_dir = seps[-2]
                new_sub_dir = os.path.join(new_dir, sub_dir)
                ensure_dir(new_sub_dir)
                new_file_path = os.path.join(new_sub_dir, name)
                shutil.copyfile(files[i], new_file_path)
            else:
                new_file_path = os.path.join(new_dir, name)
                shutil.copyfile(files[i], new_file_path)

            with open(str(new_list), 'a') as g:
                g.write(files[i])
                g.write('\n')
    else:
        for i in samples:
            with open(str(new_list), 'a') as g:
                g.write(files[i])
                g.write('\n')


def split_by_subdir(path, number, new_dir=None):
    for fn in os.listdir(path):
        new_path = os.path.join(new_dir, fn)
        ensure_dir(new_path)
        files = make_dataset_dir(os.path.join(path, fn))
        samples = random.sample(range(0, len(files)), number)
        for i in samples:
            name = os.path.split(files[i])[-1]
            new_file_path = os.path.join(new_path, name)
            shutil.copyfile(files[i], new_file_path)


if __name__ == '__main__':
    args = parser.parse_args()
    split_by_all(args.path, args.new_list, args.number, args.portion,
                 args.new_dir, args.has_subdir)

