import os
import torch.utils.data as data
from PIL import Image
from skimage import io

from .utils import check_integrity, download_and_extract_archive


class NISTDB19Dataset(data.Dataset):
    """
    NIST Special Database 19 dataset

    link: https://www.nist.gov/srd/nist-special-database-19
    """

    url = 'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip'
    md5_hash = '79572b1694a8506f2b722c7be54130c4'
    arch_name = 'by_class.zip'
    data = []
    targets = []
    folder_map = {
        'digits': {'start': 0x30, 'len': 10},
        'cap_letters': {'start': 0x41, 'len': 26},
        'low_letters': {'start': 0x61, 'len': 26},
    }

    def __init__(self, root_dir=None, train=True, download=False, data_type=None, size_limit=1000, transform=None):
        """
        Args:
            root_dir (string): Path to directory with 'by_class' folder
                (it will be created if it doesn't exist).
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            data_type (string): Data type for loading
                can be 'digits', 'cap_letters' or 'low_letters'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = os.path.expanduser(root_dir)
        self.train = train
        self.size_limit = size_limit
        self.transform = transform

        if not self.folder_map.get(data_type):
            raise RuntimeError("Argument 'data_type' should be one of: \n  " +
                               repr(tuple(self.folder_map.keys())))

        self.data_type = data_type

        if os.path.exists(self.root_dir):
            self._process(download, data_type)
        else:
            raise RuntimeError("Path '" + self.root_dir + "' doesn't exist")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[idx]

        return img, target

    def _download(self, need_to_download):
        arch_path = os.path.join(self.root_dir, self.arch_name)
        if os.path.exists(arch_path):
            if check_integrity(arch_path, self.md5_hash):
                print('Files already downloaded and verified')
                return
            else:
                ans = input('Archive corrupted. Delete and downloading it again [Y/n]? ')
                if ans in ['Y', 'y', '']:
                    os.remove(arch_path)
                    os.remove(arch_path[:len(arch_path)-4])
                else:
                    raise RuntimeError("Archive corrupted. Remove it and try again.")
        elif not need_to_download:
            raise RuntimeError("Files not found. Use 'download=True' to download them.")

        download_and_extract_archive(self.url, self.root_dir, filename=self.arch_name, md5=self.md5_hash)

    def _add_samples_from_dir(self, img_dir_path, target):
        if img_dir_path is None or target is None:
            raise RuntimeError("Image dir path and target must be valid. \nGiven path:",
                               img_dir_path, "\nGiven target:", target)
        for img_name in os.listdir(img_dir_path):
            img_path = os.path.join(img_dir_path, img_name)
            image = io.imread(img_path)
            self.data.append(image)
            self.targets.append(target)

            if len(self.data) == self.size_limit:
                return 1
        return 0

    def _process(self, download, data_type):
        self._download(download)

        start = self.folder_map[data_type]['start']
        for class_idx in range(start, start + self.folder_map[data_type]['len']):
            class_folder_name = hex(class_idx)[2:]

            if self.train:
                for hsf_idx in range(0, 8):
                    hsf_name = 'hsf_' + str(hsf_idx)
                    path_to_img_dir = os.path.join(self.root_dir,
                                                   'by_class',
                                                   class_folder_name,
                                                   hsf_name)
                    if self._add_samples_from_dir(path_to_img_dir, chr(class_idx)) == 1:
                        return
            else:
                path_to_img_dir = os.path.join(self.root_dir,
                                                'by_class',
                                               class_folder_name,
                                               'train_' + class_folder_name)
                if self._add_samples_from_dir(path_to_img_dir, chr(class_idx)) == 1:
                    return
