import torch
import torchvision
import torchvision.transforms as transforms

import alphabet_recogniser
from alphabet_recogniser.datasets import NISTDB19Dataset

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"alphabet_recogniser version: {alphabet_recogniser.__version__}\n{'#'*40}")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # ((mean1, mean2, ...), (std1, std2, ...))

train_set = NISTDB19Dataset(root_dir='./../data', data_type='low_letters', train=False, download=True,
                          transform=transform, size_limit=1000)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=0)

test_set = NISTDB19Dataset(root_dir='./../data', data_type='low_letters', train=False, download=True,
                          transform=transform, size_limit=300)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                          shuffle=True, num_workers=0)




dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))



