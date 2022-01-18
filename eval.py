import torch
import time
import os
from tqdm import tqdm
from PIL import Image

from torchvision.utils import make_grid
from utils.util import tensor2im
from utils.util import postprocess


def cal_per_frame_time(model, data_loader):
    total = 0.0
    with torch.no_grad():
        for i in range(5):
            iter_data_loader = iter(data_loader)
            start = time.time()
            for i in tqdm(range(len(data_loader))):
                image, mask, _ = next(iter_data_loader)
                image, mask = image.cuda(), mask.cuda()
                masked_img = image * mask
                pred_img = model(masked_img)
                comp_img = masked_img + (1 - mask) * pred_img
            end = time.time()
            total += (end - start) / len(data_loader.dataset)
    print("Per frame cost:", total / 5)


def save_comp_images(model, data_loader, config):
    iter_data_loader = iter(data_loader)
    with torch.no_grad():
        for batch_idx in tqdm(range(len(data_loader))):
            image, mask, img_path = next(iter_data_loader)
            image, mask = image.cuda(), mask.cuda()
            masked_img = image * mask
            pred_img = model(masked_img)
            comp_img = image * mask + (1 - mask) * pred_img

            for i in range(image.shape[0]):
                img_name = os.path.split(img_path[i])[1]
                img_name = img_name.split('.')[0]
                out_path = os.path.join(config._save_dir, img_name + '_out.png')
                img_numpy = tensor2im(comp_img[i].data)
                Image.fromarray(img_numpy).save(out_path)
    print('Finish in {}'.format(os.path.join(config._save_dir)))


def evaluate(model, data_loader, writer, logger, criterion, metric_fns):
    iter_data_loader = iter(data_loader)
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for batch_idx in tqdm(range(len(data_loader))):
            image, mask, _ = next(iter_data_loader)
            image, mask = image.cuda(), mask.cuda()
            masked_img = image * mask
            pred_img = model(masked_img)
            comp_img = image * mask + (1 - mask) * pred_img
            loss = criterion.mse_loss(pred_img, image)
            batch_size = image.shape[0]
            total_loss += loss.item() * batch_size

            comp_img = postprocess(comp_img)
            image = postprocess(image)

            # if batch_idx == 0:
            #     writer.add_image('gt', make_grid(image.cpu(), nrow=4, normalize=True))
            #     writer.add_image('masked', make_grid(masked_img.cpu(), nrow=4, normalize=True))
            #     writer.add_image('comp', make_grid(comp_img.cpu(), nrow=4, normalize=True))
            #     writer.add_image('pred', make_grid(pred_img.cpu(), nrow=4, normalize=True))
            #     break

            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(comp_img, image) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)
