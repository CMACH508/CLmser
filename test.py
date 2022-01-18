import argparse
import random
import collections
import torch
import numpy as np
from importlib import import_module
import data_loader.data_loaders as module_data
from model.loss import Loss
import evaluations.metric as module_metric
from parse_config import ConfigParser
from logger import TensorboardWriter
from eval import evaluate, save_comp_images, cal_per_frame_time

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)


def main(config):
    logger = config.get_logger('test')
    writer = TensorboardWriter(config.log_dir, logger, enabled=True)

    # setup data_loader instances
    data_loader = config.init_obj('val_data_loader', module_data, 'test')
    print(len(data_loader))

    # build model architecture, then print to console
    module_arch = import_module('model.' + config['model_name'])
    model = config.init_obj('arch', module_arch)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # get function handles of loss and metrics
    criterion = Loss()
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # calculate metrics
    evaluate(model, data_loader, writer, logger, criterion, metric_fns)

    # save qualitative results
    save_comp_images(model, data_loader, config)
    # cal_per_frame_time(model, data_loader)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--run_id', default='0', type=str,
                      help='run id (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--mn', '--model_name'], type=str, target='model_name'),
        CustomArgs(['--out', '--output_dir'], type=str, target='trainer;save_dir'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
