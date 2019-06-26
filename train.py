import os
import json
import argparse
import torch

from utils import Logger
from trainer.trainer import Trainer
import data_loader.dataloaders as module_data
import model.metric as module_metric
import model.models as module_arch
from model.loss import Loss as module_loss


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, args):
    train_logger = Logger()
    data_loader = get_instance(module_data, 'data_loader', config, 'train')
    valid_data_loader = get_instance(module_data, 'data_loader', config, 'val')

    model = get_instance(module_arch, 'model', config)

    loss = module_loss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=args.resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['model_dir'], config['name'])
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    main(config, args)
