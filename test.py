import os
import json
import argparse
import torch
from tqdm import tqdm
import data_loader.dataloaders as module_data
import model.metric as module_metric
import model.models as module_arch
from model.loss import Loss as module_loss
import numpy as np
from utils.util import _save_image, ensure_dir


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, args):
    data_loader = get_instance(module_data, 'data_loader', config, 'test')

    model = get_instance(module_arch, 'model', config)
    device = torch.device('cuda:' + str(config['gpu']))
    loss_fn = module_loss()
    model = model.to(device)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing

    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    ensure_dir(args.results_dir)
    with torch.no_grad():
        for batch_idx, gt_image in enumerate(tqdm(data_loader)):
            mask = np.ones_like(gt_image.numpy())
            # mask[:, :,  0:80, 0:50] = 0
            # mask[:, :,  0:80,0:110] = 0
            gt_image = gt_image.to(device)
            image = gt_image * torch.Tensor(mask).to(device)
            logit, fake = model(image)
            loss = loss_fn(gt_image, fake)
            batch_size = gt_image.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(gt_image, fake) * batch_size
            if batch_idx == 0:
                _save_image(args.results_dir, image, 'masked')
                _save_image(args.results_dir, gt_image, 'gt')
                _save_image(args.results_dir, fake, 'fake')

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default='trained/ex1.pth', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--results_dir', default='results', type=str,
                        help='output dictionary')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['model_dir'], config['name'])
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    main(config, args)


