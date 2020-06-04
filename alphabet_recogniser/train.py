import os, sys, time, math
import torch
import torch.optim as optim
import torchvision
import numpy as np

import alphabet_recogniser
from alphabet_recogniser.datasets.utils import NISTDB19Dataset_data_loaders
from alphabet_recogniser.models import EngAlphabetRecognizer96
from alphabet_recogniser.argparser import ArgParser
from alphabet_recogniser.test import eval_cached

from alphabet_recogniser.tensorboard_utils import upload_net_graph, save_model, add_logs_to_tensorboard, log
from alphabet_recogniser.utils import Config


get_data_loaders = NISTDB19Dataset_data_loaders


def train_network(net):
    C = Config.get_instance()
    train_loader, test_loader = get_data_loaders()

    if os.path.exists(C.path_to_model):
        net.load_state_dict(torch.load(C.path_to_model))
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        loss_values = []
        start_time = time.perf_counter()
        log_time = 0.0
        size_to_check_loss = math.floor(len(train_loader) / 5) + 1
        for epoch in range(C.epoch_num):
            net.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(C.device), data[1].to(C.device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = C.criterion(outputs, labels)
                with torch.no_grad():
                    loss_values.append(loss.item())
                    C.writer.add_scalar('Loss/train', loss.item(), C.epoch_num * epoch + i)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % size_to_check_loss == size_to_check_loss - 1 and i != 0:  # print every size_to_check_loss mini-batches
                    log('train_logs', f'[{epoch + 1}, {i + 1:3d}] loss: {running_loss / i:1.3f}')

            if (len(train_loader) - 1) % size_to_check_loss != size_to_check_loss - 1:
                log('train_logs', f'[{epoch + 1}, {len(train_loader):3d}] loss: {running_loss / (len(train_loader) - 1):1.3f}')
            log_time += add_logs_to_tensorboard(net, test_loader, epoch)
            log('train_logs',
                f'Epoch {epoch + 1}   time: {time.perf_counter() - start_time - log_time:6.0f} seconds'
                f';      log_time: {log_time:6.0f} seconds', epoch + 1)

            if C.args.m_save_period is not None and epoch % C.args.m_save_period == C.args.m_save_period - 1:
                save_model(net, f'{100 * np.mean(eval_cached(net, test_loader, epoch).TPR):3.2f}', epoch + 1)

        log_time += add_logs_to_tensorboard(net, test_loader, C.epoch_num - 1)
        log('train_logs', f'Finished Training {time.perf_counter() - start_time - log_time:6.0f} seconds')


def main():
    log('common', f"sys version: {sys.version_info}")
    log('common', f"torch version: {torch.__version__}")
    log('common', f"torchvision version: {torchvision.__version__}")
    log('common', f"alphabet_recogniser version: {alphabet_recogniser.__version__}")
    print(f"{'#' * 40}\n\n")

    C = Config.get_instance()
    train_loader, test_loader = get_data_loaders()
    net = EngAlphabetRecognizer96(num_classes=len(C.classes))
    upload_net_graph(net, test_loader)

    if C.args.classes is None:
        C.args.classes = '{'
        for el in C.classes.values():
            C.args.classes += el['chr']+','
        C.args.classes = C.args.classes[:-1] + '}'

    log('common', f'Classes: {C.args.classes[1:-1]}\n')
    net.to(C.device)
    train_network(net)

    metrics = eval_cached(net, test_loader, 0, log_loss=False)
    mean_acc_result = f'Accuracy of the network with ' + \
                      f'{len(C.classes)} classes ({C.train_size_per_class} el per class) ' + \
                      f'on {C.epoch_num} epoch: {100 * np.mean(metrics.TPR):3.2f}%'
    log('test_accuracy', mean_acc_result)
    for idx, target in enumerate(C.classes):
        log('test_accuracy_per_class', f"Recall of '{C.classes[idx]['chr']}': {100 * metrics.TPR[target]:3.2f}%")

    save_model(net, f'{100 * np.mean(metrics.TPR):3.2f}', C.epoch_num)
    C.writer.close()


if __name__ == "__main__":
    Config.get_instance().initialize(
        ArgParser.get_args()
    )
    main()
