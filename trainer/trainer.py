import torch
import numpy as np
import math
from torchvision.utils import save_image

from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.

    """

    def __init__(self, model, loss, metrics, optimizer, resume, config, data_loader,
                 valid_data_loader, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.train_logger = train_logger

    def _eval_metrics(self, output, target, split):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(split + '/' + f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _save_image(self, image, name, epoch, split):
        file_name = self.img_dir + '/{}_{}_'.format(split, epoch) + name + '.jpg'
        nrow = int(math.sqrt(self.batch_size))
        save_image(image.cpu(), file_name, nrow=nrow, padding=2, normalize=True)
        return

    def _train_epoch(self, epoch, training_loss):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, gt_image in enumerate(self.data_loader):
            gt_image = gt_image.to(self.device)
            self.optimizer.zero_grad()
            logit, fake = self.model(gt_image)
            loss = self.loss(gt_image, fake)
            loss.backward()
            self.optimizer.step()
            iter_id = (epoch - 1) * len(self.data_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), iter_id)
            training_loss.append(loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(gt_image, fake, 'train')
            if batch_idx == 0:
                self._save_image(gt_image, 'gt', epoch, 'train')
                self._save_image(fake, 'fake', epoch, 'train')
            if batch_idx % self.log_every == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log, training_loss

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, gt_image in enumerate(self.valid_data_loader):
                gt_image = gt_image.to(self.device)
                logit, fake = self.model(gt_image)
                loss = self.loss(gt_image, fake)
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(gt_image, fake, 'val')
                self.writer.add_scalar('val/total_loss', total_val_loss, epoch)
                if batch_idx == 0:
                    self._save_image(gt_image, 'gt', epoch, '  val')
                    self._save_image(fake, 'fake', epoch, 'val')
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
