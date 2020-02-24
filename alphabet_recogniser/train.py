import torch
import torchvision
import torchvision.transforms as transforms

from alphabet_recogniser.datasets import NISTDB19Dataset

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print(torch.__version__)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # ((mean1, mean2, ...), (std1, std2, ...))

testset = NISTDB19Dataset(root_dir='./../data', data_type='low_letters', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True, num_workers=0)

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
