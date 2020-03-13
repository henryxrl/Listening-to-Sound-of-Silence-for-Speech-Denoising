from utils import TrainClock
import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
from tensorboardX import SummaryWriter
import numpy as np
from networks import get_network
from visualization import draw_waveform, draw_spectrum
from transform import fast_icRM_sigmoid, istft2librosa, batch_fast_icRM_sigmoid, fast_istft
from common import PHASE_TESTING, PHASE_TRAINING
from utils import AverageMeter, TrainClock
from tqdm import tqdm

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
        self.net = self.build_net(config)

        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self, config):
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

    def record_losses(self, loss_dict, mode=PHASE_TRAINING):
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        # record loss to tensorboard
        tb = self.train_tb if mode == PHASE_TRAINING else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)

    def train_func(self, data):
        """one step of training"""
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)
        self.record_losses(losses, PHASE_TRAINING)

        return outputs, losses

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()

        with torch.no_grad():
            outputs, losses = self.forward(data)

        self.record_losses(losses, PHASE_TESTING)

        return outputs, losses

    def visualize_batch(self, data, mode, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError


class MyAgent(BaseAgent):
    def __init__(self, config):
        super(MyAgent, self).__init__(config)
        self.sr = config.sr

    def build_net(self, config):
        # customize your build_net function
        # should return the built network
        net =  get_network(config)
        # net = nn.DataParallel(net).cuda()
        # net = net.cuda()
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
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.MSELoss().cuda()

    def forward(self, data):
        # customize your forward function
        # should return the network outputs and losses
        mixed_stft = data['mixed'].cuda()
        noise_stft = data['noise'].cuda()
        clean_stft = data['clean'].cuda()  # (B, 2, 256, L)
        full_noise_stft = data['full_noise'].cuda()

        pred_noise_stft, outputs = self.net(mixed_stft, noise_stft)  # (B, 2, 256, L)

        rec_stft = batch_fast_icRM_sigmoid(mixed_stft, outputs)

        loss_stage1 = self.criterion(pred_noise_stft, full_noise_stft)
        loss_stage2 = self.criterion(rec_stft, clean_stft)
        return (pred_noise_stft, outputs), {"stage1": loss_stage1, "stage2": loss_stage2}

    def evaluate(self, dataloader):
        epoch_loss = AverageMeter("loss")
        pbar = tqdm(dataloader)
        self.net.eval()
        with torch.no_grad():
            for data in pbar:
                pbar.set_description("EVALUATION")
                outputs, losses = self.forward(data)
                epoch_loss.update(losses["stage2"].item())

        self.val_tb.add_scalar("epoch_loss", epoch_loss.avg, global_step=self.clock.epoch)

        return epoch_loss.avg

    def visualize_batch(self, data, mode, outputs=None, n=1):
        tb = self.train_tb if mode == PHASE_TRAINING else self.val_tb
        mixed_sig = data['mixed'][:n].numpy().transpose((0, 2, 3, 1))
        noise_sig = data['noise'][:n].numpy().transpose((0, 2, 3, 1))
        clean_sig = data['clean'][:n].numpy().transpose((0, 2, 3, 1))
        full_noise_sig = data['full_noise'][:n].numpy().transpose((0, 2, 3, 1))
        pred_noise_sig = outputs[0][:n].detach().cpu().numpy().transpose((0, 2, 3, 1))
        output_mask = outputs[1][:n].detach().cpu().numpy().transpose((0, 2, 3, 1))
        # groups = np.concatenate([mixed_sig, noise_sig, clean_sig, output_sig], axis=1)  # (n, 4, len)
        for i in range(n):
            output_sig = fast_icRM_sigmoid(mixed_sig[i], output_mask[i])
            # wavefrom = draw_waveform([mixed_sig[i], noise_sig[i], clean_sig[i], output_sig[i]])
            spectrum = draw_spectrum([fast_istft(mixed_sig[i]),
                                      fast_istft(noise_sig[i]),
                                      fast_istft(full_noise_sig[i]),
                                      fast_istft(pred_noise_sig[i]),
                                      fast_istft(clean_sig[i]),
                                      fast_istft(output_sig)])
            # wavefrom = wavefrom.transpose(2, 0, 1)[::-1]
            spectrum = spectrum.transpose(2, 0, 1)[::-1]

            # tb.add_image('waveform_{}'.format(i), wavefrom, global_step=self.clock.step)
            tb.add_image('spectrum_{}'.format(i), spectrum, global_step=self.clock.step)

            # tb.add_audio('mixed_{}'.format(i), mixed_sig[i:i+1], global_step=self.clock.step, sample_rate=self.sr)
            # tb.add_audio('noise_{}'.format(i), noise_sig[i:i+1], global_step=self.clock.step, sample_rate=self.sr)
            # tb.add_audio('clean_{}'.format(i), clean_sig[i:i+1], global_step=self.clock.step, sample_rate=self.sr)
            # tb.add_audio('output_{}'.format(i), output_sig[i:i+1], global_step=self.clock.step, sample_rate=self.sr)
