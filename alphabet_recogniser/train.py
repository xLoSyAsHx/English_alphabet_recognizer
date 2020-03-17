import os
import sys
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


print(f"sys version: {sys.version_info}")
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

if args.use_preprocessed_data:
    NISTDB19Dataset.download_and_preprocess(root_dir=args.root_dir, data_type=args.data_type, str_classes=args.classes)

train_set = NISTDB19Dataset(root_dir=args.root_dir, data_type=args.data_type, train=True, download=True,
                            str_classes=args.classes, use_preproc=args.use_preprocessed_data,
                            transform=transform, size_limit=args.train_limit)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                           shuffle=args.shuffle_train, num_workers=0)

test_set = NISTDB19Dataset(root_dir=args.root_dir, data_type=args.data_type, train=False, download=True,
                           str_classes=args.classes, use_preproc=args.use_preprocessed_data,
                           transform=transform, size_limit=args.test_limit)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=50,
                                          shuffle=args.shuffle_test, num_workers=0)

net = EngAlphabetRecognizer96(num_classes=len(train_set.classes))

device = torch.device('cpu')
if torch.cuda.is_available() and False:
    print('\nCuda is available\n\n')
    device = torch.device('cuda')
    net.to(device)
else:
    print('\nCuda is unavailable\n\n')

PATH = 'cifar_net.torchmodel'

EPOCH_NUM = args.e
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
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / i:1.3f}')
        print(f'Epoch {epoch}   time: {time.perf_counter() - start_time:6.0f} seconds')

    print(f'Finished Training {time.perf_counter() - start_time:6.0f} seconds')
    x = np.arange(0, len(loss_values))
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.grid()
    ax.set(xlabel='batch №. (batch_size=50)', ylabel='loss',
           title='Loss')
    ax.plot(x, loss_values)
    fig.savefig(f"./../data/loss_e[{EPOCH_NUM}]_cl[{len(train_set.classes)}]_tr_s[{train_set.size_per_class}].png", dpi=400)
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
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        total += labels.size(0)
        correct += (p_indexes == labels).sum().item()

mean_acc_result = f'Accuracy of the network with ' + \
                  f'{len(test_set.classes)} classes ({train_set.size_per_class} el per class) ' + \
                  f'on {EPOCH_NUM} epoch: {100 * correct / total:3.2f}%'
print(mean_acc_result)

with open(f'./../data/stat_e[{EPOCH_NUM}]_cl[{len(train_set.classes)}]_tr_s[{train_set.size_per_class}].txt',
          'wb') as stat_file:
    b = bytearray()
    b.extend(map(ord, mean_acc_result))
    stat_file.write(b)

    for idx, target in enumerate(test_set.classes):
        symbol = chr(NISTDB19Dataset.folder_map['low_letters']['start'] + target)
        s = f"Accuracy of '{test_set.classes[idx]['chr']}': {100 * class_correct[target] / class_total[target]:3.2f}%"
        b = bytearray()
        b.extend(map(ord, s))
        stat_file.write(b)
        print(s)
