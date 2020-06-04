import sys
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from alphabet_recogniser.datasets import NISTDB19Dataset


class Config:
    __instance__ = None
    __is_initialized__ = False

    @staticmethod
    def get_instance():
        if Config.__instance__ is None:
            Config()
        return Config.__instance__

    def __init__(self):
        if Config.__instance__ is not None:
            raise Exception("This class is a singleton!")
        else:
            self.args = None
            self.writer = None
            self.log_pref = None

            self.train_transform = None
            self.test_transform = None
            self.classes = None
            self.train_size_per_class = None
            self.test_size_per_class = None

            self.device = None
            self.criterion = None
            self.epoch_num = None
            self.path_to_model = None

            Config.__instance__ = self

    def initialize(self, args):
        self = Config.__instance__
        if Config.__is_initialized__:
            raise Exception("Config was already initialized")

        self.train_transform = transforms.Compose(
            [transforms.CenterCrop(112),
             transforms.RandomCrop(96),
             transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

        self.test_transform = transforms.Compose(
            [transforms.CenterCrop(96),
             transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

        self.args = args
        self.train_size_per_class = self.args.train_limit
        self.test_size_per_class = self.args.test_limit

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_num = self.args.e
        self.path_to_model = self.args.m_load_path

        if self.args.t_cm_freq is None:
            self.args.t_cm_freq = sys.maxsize

        if self.args.t_precision_bar_freq is None:
            self.args.t_precision_bar_freq = sys.maxsize

        num_classes = len(self.args.classes[1:-1].split(',')) if self.args.classes is not None \
                 else NISTDB19Dataset.folder_map[self.args.data_type]['len']

        self.log_pref = datetime.now().strftime('%Y_%B%d_%H-%M-%S')
        self.writer = SummaryWriter(log_dir=f"{self.args.t_logdir}{self.log_pref}"
                                         f"_e[{self.epoch_num}]"
                                         f"_c[{num_classes}]"
                                         f"_tr_s[{self.train_size_per_class if self.train_size_per_class is not None else 'All'}]"
                                         f"_t_s[{self.test_size_per_class if self.test_size_per_class is not None else 'All'}]")

        Config.is_initialized = True


def imshow(img):
    img = img / 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
