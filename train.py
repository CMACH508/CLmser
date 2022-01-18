import argparse
import random
import collections
import torch
import numpy as np
from importlib import import_module
from trainer import Trainer
import data_loader.data_loaders as module_data
from model.loss import Loss
import evaluations.metric as module_metric
from parse_config import ConfigParser

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, 'train')
    print("training images = %d" % len(data_loader.dataset))
    valid_data_loader = config.init_obj('val_data_loader', module_data, 'test')

    # build model architecture, then print to console
    module_arch = import_module('model.' + config['model_name'])
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # build optimizer, learning rate scheduler
    optimiser = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimiser)

    # get function handles of loss and metrics
    criterion = Loss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainer = Trainer(model, criterion, metrics, optimiser,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


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
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--mn', '--model_name'], type=str, target='model_name'),
        CustomArgs(['--dn', '--dataset_name'], type=str, target='dataset'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
