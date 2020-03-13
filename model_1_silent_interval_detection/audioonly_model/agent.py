import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# python3 -m pip install --user tensorboardX
from tensorboardX import SummaryWriter
# python3 -m pip install --user tqdm
from tqdm import tqdm

from networks import get_network
from utils import AverageMeter, TrainClock
from tools import weighted_binary_cross_entropy

# from common import PRETRAINED_MODEL_PATH
# from networks import set_requires_grad


def get_agent(config):
    return MyAgent(config)


class BaseAgent(object):
    """Base trainer that provides commom training behavior. 
        All trainer should be subclass of this class.
    """
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.device = config.device
        self.batch_size = config.batch_size

        # build network
        self.net = self.build_net()

        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self):
        raise NotImplementedError

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.MSELoss().to(self.device)

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Checkpoint saved at {}".format(save_path))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if isinstance(self.net, nn.DataParallel):
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)
        else:
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': self.net.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)
        self.net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Checkpoint loaded from {}".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    @abstractmethod
    def forward(self, data):
        pass

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        self.scheduler.step(self.clock.epoch)

    def record_losses(self, loss_dict, mode='train'):
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        # record loss to tensorboard
        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)

    def train_func(self, data):
        """one step of training"""
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)
        self.record_losses(losses, 'train')

        return outputs, losses

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()

        with torch.no_grad():
            outputs, losses = self.forward(data)

        self.record_losses(losses, 'validation')

        return outputs, losses

    def visualize_batch(self, data, mode, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError


class MyAgent(BaseAgent):
    def __init__(self, config):
        super(MyAgent, self).__init__(config)

    def build_net(self):
        # customize your build_net function
        # should return the built network
        net = get_network()
        # print('Loading pretrained model "{}"...'.format(PRETRAINED_MODEL_PATH))
        # net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH), strict=False)
        # Fix all the layers till the last one
        # net.eval()
        # set_requires_grad(net, False)
        # set_requires_grad(net.my_logits, True)
        if torch.cuda.device_count() > 1:
            print('Multi-GPUs available')
            net = nn.DataParallel(net.cuda())   # For multi-GPU
        else:
            print('Single-GPU available')
            net = net.cuda()    # For single-GPU
        return net

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        # only train the last layer
        # if torch.cuda.device_count() > 1:
        #     self.optimizer = optim.Adam(self.net.module.my_logits.parameters(), config.lr)
        # else:
        #     self.optimizer = optim.Adam(self.net.my_logits.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.BCEWithLogitsLoss().cuda()

    def forward(self, data):
        # customize your forward function
        # should return the network outputs and losses
        # frames = data['frames'].cuda()
        label = data['label'].cuda()
        audio = data['audio'].cuda()
        # print(label.size())
        # weights = data['weights'].cuda()
        # print(weights.size())

        output = self.net(audio)
        # print(output.size())
        # print('output:', output)
        loss = self.criterion(output, label)
        # loss = weighted_binary_cross_entropy(output, label, weights)
        # print('loss', loss)

        return output, {"bce": loss}

    def evaluate(self, dataloader):
        metrics = AverageMeter("loss")
        epoch_acc = AverageMeter("acc")
        pbar = tqdm(dataloader)
        self.net.eval()
        with torch.no_grad():
            for data in pbar:
                pbar.set_description("EVALUATION")
                outputs, losses = self.forward(data)
                # print('outputs:', outputs)
                # print('losses:', losses)
                metrics.update(losses["bce"].item())

                pred_labels = (torch.sigmoid(outputs).detach().cpu().numpy() >= 0.5).astype(np.float)
                # print('pred_labels:', pred_labels)
                labels = data['label'].numpy()
                # print('labels:', labels)
                acc = np.mean(pred_labels == labels)
                # print('acc:', acc)
                epoch_acc.update(acc)

        self.val_tb.add_scalar("epoch_loss", metrics.avg, global_step=self.clock.epoch)
        self.val_tb.add_scalar("epoch_acc", epoch_acc.avg, global_step=self.clock.epoch)

        return epoch_acc.avg

    # def visualize_batch(self, data, mode, outputs=None, n=1):
    #     tb = self.train_tb if mode == 'train' else self.val_tb
    #     for i in range(n):
    #         tb.add_image('input_{}'.format(i), data['frames'][i], global_step=self.clock.step)
