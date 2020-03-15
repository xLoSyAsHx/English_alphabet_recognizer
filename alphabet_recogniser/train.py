import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import alphabet_recogniser
from alphabet_recogniser.datasets import NISTDB19Dataset
from alphabet_recogniser.models import EngAlphabetRecognizer96
from alphabet_recogniser.argparser import ArgParser

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"alphabet_recogniser version: {alphabet_recogniser.__version__}\n{'#' * 40}")

transform = transforms.Compose(
    [transforms.CenterCrop(112),
     transforms.RandomCrop(96),
     transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))
     ])

if __name__ == "__main__":
    parser = ArgParser.get_instance()
    args = parser.parse_args()
    parser.check_compatibility(args)

NISTDB19Dataset.download_and_preprocess(root_dir='./../data', data_type='low_letters')

train_set = NISTDB19Dataset(root_dir='./../data', data_type='low_letters', train=True, download=True,
                            use_preproc=True,
                            transform=transform, size_limit=100, size_limit_per_class=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=40,
                                           shuffle=True, num_workers=0)

test_set = NISTDB19Dataset(root_dir='./../data', data_type='low_letters', train=False, download=True,
                           use_preproc=True,
                           transform=transform, size_limit=300, size_limit_per_class=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=40,
                                          shuffle=True, num_workers=0)

net = EngAlphabetRecognizer96(num_classes=26)

device = torch.device('cpu')
if torch.cuda.is_available() and False:
    print('\nCuda is available\n\n')
    device = torch.device('cuda')
    net.to(device)
else:
    print('\nCuda is unavailable\n\n')

PATH = 'cifar_net.torchmodel'

EPOCH_NUM = 4
if os.path.exists(PATH) and False:
    net.load_state_dict(torch.load(PATH))
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_values = []
    start_time = time.perf_counter()
    for epoch in range(EPOCH_NUM):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            with torch.no_grad():
                loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / i))
        print('Epoch', epoch, '   time:', time.perf_counter() - start_time, 'seconds')

    print('Finished Training ', time.perf_counter() - start_time, 'seconds')
    x = np.arange(0, len(loss_values))
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.grid()
    ax.set(xlabel='batch №. (batch_size=40)', ylabel='loss',
           title='Loss')
    ax.plot(x, loss_values)
    fig.savefig(f"./../data/loss_e[{EPOCH_NUM}]_cl[{len(train_set.classes)}]_tr_s[{1000}].png", dpi=400)
    step = round(len(loss_values) / (train_set.size_per_class / train_loader.batch_size))

correct = 0
total = 0
class_correct = np.zeros(len(train_set.classes), dtype=int)
class_total = np.zeros(len(train_set.classes), dtype=int)
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)
        p_values, p_indexes = torch.max(outputs.data, 1)
        c = (p_indexes == labels).squeeze()
        for i in range(40):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        total += labels.size(0)
        correct += (p_indexes == labels).sum().item()

print(f'Accuracy of the network with {EPOCH_NUM} epoch: {100 * correct / total}')

with open(f'./../data/stat_e[{EPOCH_NUM}]_cl[{len(train_set.classes)}]_tr_s[{1000}].txt', 'wb') as stat_file:
    s = f'Accuracy of the network with {EPOCH_NUM} epoch: {100 * correct / total}'
    b = bytearray()
    b.extend(map(ord, s))
    stat_file.write(b)

    for target in test_set.classes:
        symbol = chr(NISTDB19Dataset.folder_map['low_letters']['start'] + target)
        s = f"Accuracy of '{symbol}': {100 * class_correct[target] / class_total[target]}"
        b = bytearray()
        b.extend(map(ord, s))
        stat_file.write(b)
        print(s)
