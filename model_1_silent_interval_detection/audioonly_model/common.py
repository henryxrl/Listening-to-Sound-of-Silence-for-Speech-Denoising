import multiprocessing
import os
from abc import abstractmethod

import torch

import utils


PROJECT_ROOT = "../"
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "model_output")
EXPERIMENT_NAME = os.path.basename(os.getcwd())
EXPERIMENT_DIR = os.path.join(OUTPUT_ROOT, EXPERIMENT_NAME)
PHASE_TRAINING = 'training'
PHASE_TESTING = 'testing'
PHASE_PREDICTION = 'pred'

# PRETRAINED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'pretrained'))
# MODEL = 'rgb_charades.pt'
# PRETRAINED_MODEL_PATH = os.path.join(PRETRAINED_DIR, MODEL)


def get_config(args):
    config = MyConfig()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    config.device = torch.device("cuda:{}".format(args.gpu_ids))
    return config


class Config(object):
    """Base class of Config, provide necessary hyperparameters."""
    def __init__(self):
        # general
        self.device = None
        # self.sr = 16000
        # self.clip_len = 32768

        # experiment paths
        self.proj_dir = OUTPUT_ROOT
        self.exp_name = EXPERIMENT_NAME
        self.exp_dir = EXPERIMENT_DIR

        # create soft link to experiment log directory
        if not os.path.exists('train_log'):
            os.symlink(self.exp_dir, 'train_log')

        self.log_dir, self.model_dir = self.set_exp_paths()
        utils.ensure_dirs([self.log_dir, self.model_dir])

        # network configuration
        self.netType = 'context'
        self.set_network_info()

        # training configuration
        self.nr_epochs = 100
        self.batch_size = 15    # GPU memory usage
        self.num_workers = 60 #32 #multiprocessing.cpu_count()    # RAM usage
        self.lr = 1e-3 #1e-4
        self.lr_step_size = 15
        self.lr_decay = 0.999

        self.save_frequency = 1
        self.val_frequency = 10
        self.visualize_frequency = 100

        self.points_batch_size = None

    def __repr__(self):
        return "epochs: {}\nbatch size: {}\nlr: {}\nworkers: {}\ndevice: {}\n".format(
            self.nr_epochs, self.batch_size, self.lr, self.num_workers, self.device
        )

    def set_exp_paths(self):
        return os.path.join(self.exp_dir, 'log'), os.path.join(self.exp_dir, 'model')

    def set_network_info(self):
        raise NotImplementedError


class MyConfig(Config):
    def set_network_info(self):
        # customize your set_network_info function
        # should set hyperparameters for network architecture
        # self.kernel_sizes = [(1, 7), (7, 1), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]
        # self.dilations    = [(1, 1), (1, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
        # self.en_channels = [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024]
        # self.de_channels = [x * 2 for x in list(reversed(self.en_channels))[:-1]] + [1]
        pass
