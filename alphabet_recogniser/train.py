import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from datetime import datetime

import alphabet_recogniser
from alphabet_recogniser.datasets import NISTDB19Dataset
from alphabet_recogniser.models import EngAlphabetRecognizer96
from alphabet_recogniser.argparser import ArgParser

from alphabet_recogniser.utils import calculate_metrics, log_conf_matrix, log_TPR_PPV_F1_bars, log_ROC_AUC


class Globals:
    def __init__(self):
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
G = Globals()


def log(tag, text, glogal_step=0):
    print(text)
    G.writer.add_text(tag, text, glogal_step)


def setup_global_vars():
    G.train_size_per_class = G.args.train_limit
    G.test_size_per_class = G.args.test_limit

    G.criterion = nn.CrossEntropyLoss()
    G.device = torch.device('cuda')
    G.epoch_num = G.args.e
    G.path_to_model = 'cifar_net.torchmodel'

    if G.args.t_cm_granularity is None:
        G.args.t_cm_granularity = sys.maxsize

    if G.args.t_precision_bar_gran is None:
        G.args.t_precision_bar_gran = sys.maxsize

    num_classes = len(G.args.classes[1:-1].split(',')) if G.args.classes is not None \
             else NISTDB19Dataset.folder_map[G.args.data_type]['len']

    G.log_pref = datetime.now().strftime('%Y_%B%d_%H-%M-%S')
    G.writer = SummaryWriter(log_dir=f"{G.args.t_logdir}{G.log_pref}"
                                     f"_e[{G.epoch_num}]"
                                     f"_c[{num_classes}]"
                                     f"_tr_s[{G.train_size_per_class if G.train_size_per_class is not None else 'All'}]"
                                     f"_t_s[{G.test_size_per_class if G.test_size_per_class is not None else 'All'}]")


def get_data_loaders(force_shuffle_test=False):
    if hasattr(get_data_loaders, 'train'):
        return get_data_loaders.train, get_data_loaders.test

    if G.args.use_preprocessed_data:
        NISTDB19Dataset.download_and_preprocess(root_dir=G.args.root_dir, data_type=G.args.data_type,
                                                str_classes=G.args.classes)

    if G.args.train_path in None:
        train_set = NISTDB19Dataset(root_dir=G.args.root_dir, data_type=G.args.data_type, train=True, download=True,
                                    str_classes=G.args.classes, use_preproc=G.args.use_preprocessed_data,
                                    train_transform=G.train_transform, test_transform=G.test_transform,
                                    size_limit=G.args.train_limit)
    else:
        train_set = NISTDB19Dataset.load_from_file(G.args.train_path)
    get_data_loaders.train = DataLoader(train_set, batch_size=G.args.batch_size,
                                        shuffle=G.args.shuffle_train, num_workers=0)

    if G.args.test_path in None:
        test_set = NISTDB19Dataset(root_dir=G.args.root_dir, data_type=G.args.data_type, train=False, download=True,
                                   str_classes=G.args.classes, use_preproc=G.args.use_preprocessed_data,
                                   train_transform=G.train_transform, test_transform=G.test_transform,
                                   size_limit=G.args.test_limit)
    else:
        test_set = NISTDB19Dataset.load_from_file(G.args.test_path)
    get_data_loaders.test = DataLoader(test_set, batch_size=G.args.batch_size,
                                       shuffle=G.args.shuffle_test if force_shuffle_test is False else True,
                                       num_workers=0)

    G.classes = train_set.classes
    return get_data_loaders.train, get_data_loaders.test


def upload_net_graph(net):
    if G.args.t_images is None:
        with torch.no_grad():
            net.to('cpu')
            G.writer.add_graph(net)
            net.to(G.device)
            return

    with torch.no_grad():
        _, test_loader = get_data_loaders(force_shuffle_test=True)
        images = iter(test_loader).next()[0][:G.args.t_images]
        images = images.to('cpu')
        net.to('cpu')
        G.writer.add_image('MNIST19 preprocessed samples', torchvision.utils.make_grid(images))
        G.writer.add_graph(net, images)
        net.to(G.device)


def add_logs_to_tensorboard(net, epoch):
    start_time = time.perf_counter()
    _, test_loader = get_data_loaders()
    classes = [G.classes[key]['chr'] for key in G.classes]
    metrics = calculate_metrics(G, net, test_loader, epoch)

    def can_log(granularity):
        return (epoch % granularity == 0 and epoch != 0) or (epoch % granularity != 0 and G.epoch_num - 1 == epoch)

    if can_log(G.args.t_cm_granularity):
        log_conf_matrix(G, metrics, classes, epoch)

    if can_log(G.args.t_precision_bar_gran):
        log_TPR_PPV_F1_bars(G, metrics, classes, epoch)

    if can_log(G.args.t_roc_auc_gran):
        log_ROC_AUC(G, metrics, classes, epoch)

    return time.perf_counter() - start_time


def train_network(net):
    train_loader, test_loader = get_data_loaders()

    if os.path.exists(G.path_to_model) and False:
        net.load_state_dict(torch.load(G.path_to_model))
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        loss_values = []
        start_time = time.perf_counter()
        log_time = 0.0
        size_to_check_loss = math.floor(len(train_loader) / 5) + 1
        for epoch in range(G.epoch_num):
            net.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(G.device), data[1].to(G.device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = G.criterion(outputs, labels)
                with torch.no_grad():
                    loss_values.append(loss.item())
                    G.writer.add_scalar('Loss/train', loss.item(), G.epoch_num * epoch + i)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % size_to_check_loss == size_to_check_loss - 1 and i != 0:  # print every size_to_check_loss mini-batches
                    print(f'[{epoch + 1}, {i + 1:3d}] loss: {running_loss / i:1.3f}')

            if (len(train_loader) - 1) % size_to_check_loss != size_to_check_loss - 1:
                print(f'[{epoch + 1}, {len(train_loader):3d}] loss: {running_loss / (len(train_loader) - 1):1.3f}')
            log('train_logs', f'Epoch {epoch + 1}   time: {time.perf_counter() - start_time - log_time:6.0f} seconds', epoch + 1)
            log_time += add_logs_to_tensorboard(net, epoch)

        log_time += add_logs_to_tensorboard(net, G.epoch_num - 1)
        log('train_logs', f'Finished Training {time.perf_counter() - start_time - log_time:6.0f} seconds')


def save_model(net, acc):
    if G.args.m_save_path is None:
        return

    torch.save(net,os.path.join(
                       G.args.m_save_path,
                       f"{G.log_pref}"
                       f"_acc[{acc}]"
                       f"_e[{G.epoch_num}]"
                       f"_c[{net.num_classes}]"
                       f"_tr_s[{G.train_size_per_class if G.train_size_per_class is not None else 'All'}]"
                       f"_t_s[{G.test_size_per_class if G.test_size_per_class is not None else 'All'}]"
                       f".model"))


def main():
    parser = ArgParser.get_instance()
    G.args = parser.parse_args()
    parser.check_compatibility(G.args)

    setup_global_vars()  # All, except transform
    G.train_transform = transforms.Compose(
        [transforms.CenterCrop(112),
         transforms.RandomCrop(96),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    G.test_transform = transforms.Compose(
        [transforms.CenterCrop(112),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    log('common', f"sys version: {sys.version_info}")
    log('common', f"torch version: {torch.__version__}")
    log('common', f"torchvision version: {torchvision.__version__}")
    log('common', f"alphabet_recogniser version: {alphabet_recogniser.__version__}")
    print(f"{'#' * 40}\n\n")

    train_loader, test_loader = get_data_loaders()
    net = EngAlphabetRecognizer96(num_classes=len(G.classes))

    if torch.cuda.is_available():
        log('common', 'Cuda is available\n')
    else:
        log('common', 'Cuda is unavailable\n')
        G.device = torch.device('cpu')

    if G.args.classes is None:
        G.args.classes = '{'
        for el in G.classes.values():
            G.args.classes += el['chr']+','
        G.args.classes = G.args.classes[:-1] + '}'

    log('common', f'Classes: {G.args.classes[1:-1]}\n')
    net.to(G.device)
    train_network(net)
    upload_net_graph(net)

    metrics = calculate_metrics(G, net, test_loader, 0, log_loss=False)
    mean_acc_result = f'Accuracy of the network with ' + \
                      f'{len(G.classes)} classes ({G.train_size_per_class} el per class) ' + \
                      f'on {G.epoch_num} epoch: {100 * np.mean(metrics.TPR):3.2f}%'
    log('test_accuracy', mean_acc_result)
    for idx, target in enumerate(G.classes):
        log('test_accuracy_per_class', f"Recall of '{G.classes[idx]['chr']}': {100 * metrics.TPR[target]:3.2f}%")

    save_model(net, f'{100 * np.mean(metrics.TPR):3.2f}')
    G.writer.close()


if __name__ == "__main__":
    main()
