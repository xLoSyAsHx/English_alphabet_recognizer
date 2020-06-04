import os, sys, json
import compress_pickle
import numpy as np
import torch.utils.data as data

from PIL import Image
from skimage import io
from tqdm import tqdm

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, calculate_md5


class NISTDB19Dataset(data.Dataset):
    """
    NIST Special Database 19 dataset

    link: https://www.nist.gov/srd/nist-special-database-19
    """

    url = 'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip'
    md5_hash = '79572b1694a8506f2b722c7be54130c4'
    arch_name = 'by_class.zip'
    folder_map = {
        'digits': {'start': 0x30, 'len': 10},
        'cap_letters': {'start': 0x41, 'len': 26},
        'low_letters': {'start': 0x61, 'len': 26},
    }

    class Sample:
        def __init__(self, image, target):
            if image.shape != (128, 128, 3):
                raise RuntimeError(f"Invalid image shape. Got {image.shape}. Expected {(128, 128, 3)}")

            self.image = image
            self.target = target

    def __init__(self, root_dir=None, train=True, download=False, data_type=None,
                 size_limit=0, train_transform=None, test_transform=None,
                 str_classes=None, use_preproc=False, verify=True):
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
            size_limit (int): Dataset size limit per class
            train_transform (callable, optional): Optional transform to be applied
                on a sample for train.
            test_transform (callable, optional): Optional transform to be applied
                on a sample for test.
            str_classes (str, optional): Specify class to upload. Example: '{a,b,c}'
            use_preproc (bool, optional): Use by_class_preproc dir for data uploading
            verify (bool, optional): If false - md5 verification will be skipped
        """

        if not self.folder_map.get(data_type):
            raise RuntimeError("Argument 'data_type' should be one of: \n  " +
                               repr(tuple(self.folder_map.keys())))
        self.data = []
        self.targets = []
        self.classes = {}
        self.data_type = data_type
        self.root_dir = os.path.expanduser(root_dir)
        self.train = train
        self.train_transform = train_transform
        self.test_transform  = test_transform
        self.size_per_class = size_limit

        if os.path.exists(self.root_dir):
            print(f"\nLoad {'train' if self.train else 'test'} dataset:")
            self._process(
                download,
                data_type,
                str_classes[1:-1].split(',') if str_classes is not None else None,
                verify,
                use_preproc)
        else:
            raise RuntimeError("Path '" + self.root_dir + "' doesn't exist")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.train_transform is not None and self.train:
            img = self.train_transform(img)
        elif self.test_transform is not None and not self.train:
            img = self.test_transform(img)
        target = self.targets[idx]

        return img, target

    @staticmethod
    def __zip_folder_to_batches__(sourse_dirs, target, size_per_batch):

        idx = 0
        batches = [np.ndarray(size_per_batch, dtype=NISTDB19Dataset.Sample)]
        for dir_path in sourse_dirs:
            for filename in os.listdir(dir_path):
                if idx == size_per_batch - 1:
                    batches.append(np.ndarray(size_per_batch, dtype=NISTDB19Dataset.Sample))
                    idx = 0

                img_path = os.path.join(dir_path, filename)
                image = io.imread(img_path)
                batches[-1][idx] = NISTDB19Dataset.Sample(image, target)
                idx += 1

        return batches

    @staticmethod
    def __save_batches__(batches, path_to_batches, prefix, compression='gzip'):
        manifest_path = os.path.join(path_to_batches, 'manifest.json')
        if not os.path.exists(manifest_path):
            data = {}
            with open(manifest_path, 'w') as manifest_file:
                json.dump(data, manifest_file)

        with open(manifest_path, 'r') as manifest_file:
            manifest = json.load(manifest_file)

        for i, batch in enumerate(batches):
            batch_name = f"{prefix}_{i:04d}.batch"
            if batch_name in manifest:
                continue

            with open(os.path.join(path_to_batches, batch_name), 'wb') as batch_file:
                compress_pickle.dump(batch, batch_file, compression=compression, protocol=4)

            md5 = calculate_md5(os.path.join(path_to_batches, batch_name))
            manifest[batch_name] = {'md5': md5, 'compression': compression}

        with open(manifest_path, 'w') as manifest_file:
            json.dump(manifest, manifest_file)

    @staticmethod
    def __load_batch__(batch_name, path_to_batches, check_md5=True):
        manifest_path = os.path.join(path_to_batches, 'manifest.json')
        if not os.path.exists(manifest_path):
            raise RuntimeError(f'Manifest does not found in {manifest_path}')

        with open(manifest_path, 'r') as manifest_file:
            manifest = json.load(manifest_file)

        if batch_name not in manifest:
            raise RuntimeError(f'{batch_name} does not found in manifest {manifest_path}')

        if check_md5 and not check_integrity(os.path.join(path_to_batches, batch_name),
                                             md5=manifest[batch_name]['md5']):
            raise RuntimeError(f'{batch_name} md5 mismatch')

        compression = manifest[batch_name]['compression']
        with open(os.path.join(path_to_batches, batch_name), 'rb') as batch_file:
            return compress_pickle.load(batch_file, compression=compression)

    @staticmethod
    def download_and_preprocess(root_dir, data_type, str_classes=None, check_md5=True):
        print(f"Download and preprocess '{data_type}':")
        DS = NISTDB19Dataset
        DS.download(root_dir, True, check_md5)

        preproc_pdath = os.path.join(root_dir, 'by_class_preproc')
        if not os.path.exists(preproc_pdath):
            os.mkdir(preproc_pdath)

        arr_classes = str_classes[1:-1].split(',') if str_classes is not None else None
        start = DS.folder_map[data_type]['start']
        ds_len = DS.folder_map[data_type]['len'] if arr_classes is None else len(arr_classes)
        idx_range = range(start, start + ds_len) if arr_classes is None else [ord(x) for x in arr_classes]
        for class_idx in tqdm(idx_range,
                              file=sys.stdout,
                              bar_format='{l_bar}{' + f'bar:{ds_len * 2}' + '}{r_bar}{' + f'bar:-{ds_len * 2}b' + '}'):

            path_to_img_dirs = os.path.join(root_dir, 'by_class', hex(class_idx)[2:])
            path_to_batches = os.path.join(preproc_pdath, hex(class_idx)[2:])

            if os.path.exists(path_to_batches):
                continue

            source_dirs = [os.path.join(path_to_img_dirs, x) for x in os.listdir(path_to_img_dirs)]
            train_batches = DS.__zip_folder_to_batches__(
                [x for x in source_dirs if os.path.isdir(x) and 'train' in x],
                class_idx - start,
                8000)
            test_batches = DS.__zip_folder_to_batches__(
                [x for x in source_dirs if os.path.isdir(x) and 'train' not in x],
                class_idx - start,
                8000)

            if not os.path.exists(path_to_batches):
                os.mkdir(path_to_batches)

            DS.__save_batches__(train_batches, path_to_batches, prefix='train')
            DS.__save_batches__(test_batches, path_to_batches, prefix='test')

    @staticmethod
    def save_to_file(dataset, filepath, force_overwrite=False, compression='lzma'):
        if not isinstance(dataset, NISTDB19Dataset):
            raise RuntimeError(f"Object {type(dataset)} is not inherit from NISTDB19Dataset")

        if not os.path.exists(os.path.dirname(filepath)):
            raise RuntimeError(f"Folder {os.path.dirname(filepath)} does not exist")

        if not force_overwrite and os.path.exists(filepath):
            print(
                f"\n[WARNING]: Can't save to '{filepath}'.\nFile already exist. Add 'force_overwrite=False' for overwrite")
            return
        with open(filepath, 'wb') as dataset_file:
            compress_pickle.dump(dataset, dataset_file, compression=compression)

    @staticmethod
    def load_from_file(filepath, compression='lzma'):
        if not os.path.exists(filepath):
            raise RuntimeError(f"Can't open {filepath}. File doesn't exist")
        with open(filepath, 'rb') as dataset_file:
            return compress_pickle.load(dataset_file, compression=compression)

    @staticmethod
    def download(root_dir, download, verify):
        arch_path = os.path.join(root_dir, NISTDB19Dataset.arch_name)
        if os.path.exists(arch_path):
            if not verify:
                print('Verification was skipped', flush=True)
                return

            if check_integrity(arch_path, NISTDB19Dataset.md5_hash):
                print('Files already downloaded and verified', flush=True)
                return
            else:
                ans = input('Archive corrupted. Delete and downloading it again [Y/n]? ')
                if ans in ['Y', 'y', '']:
                    os.remove(arch_path)
                    os.remove(arch_path[:len(arch_path) - 4])
                else:
                    raise RuntimeError("Archive corrupted. Remove it and try again.")
        elif not download:
            raise RuntimeError("Files not found. Use 'download=True' to download them.")

        download_and_extract_archive(NISTDB19Dataset.url, root_dir,
                                     filename=NISTDB19Dataset.arch_name,
                                     md5=NISTDB19Dataset.md5_hash)

    def _add_samples_from_dir(self, img_dir_path, target, newly_added):
        if img_dir_path is None or target is None:
            raise RuntimeError("Image dir path and target must be valid. \nGiven path:",
                               img_dir_path, "\nGiven target:", target)

        for img_name in os.listdir(img_dir_path):
            img_path = os.path.join(img_dir_path, img_name)
            image = io.imread(img_path)
            self.data.append(image)
            self.targets.append(target)
            newly_added += 1

            if self.size_per_class != 0 and newly_added == self.size_per_class:
                return 1
        return 0

    def _add_samples_from_batches(self, img_dir_path, prefix, target, check_md5):
        if img_dir_path is None:
            raise RuntimeError(f'Image dir path and target must be valid. \nGiven path: {img_dir_path}')

        newly_added = 0
        batch_names = os.listdir(img_dir_path)
        batch_names.remove('manifest.json')
        for batch_name in [x for x in batch_names if prefix in x]:
            batch = self.__load_batch__(batch_name, img_dir_path, check_md5)

            for sample in batch:
                if sample is None:
                    return

                self.data.append(sample.image)
                self.targets.append(target)
                newly_added += 1

                if self.size_per_class != 0 and newly_added == self.size_per_class:
                    return

    def _process(self, download, data_type, arr_classes, verify, use_preproc):
        NISTDB19Dataset.download(self.root_dir, download, verify)

        start = self.folder_map[data_type]['start']
        ds_len = NISTDB19Dataset.folder_map[data_type]['len'] if arr_classes is None else len(arr_classes)
        idx_range = range(start, start + ds_len) if arr_classes is None else [ord(x) for x in arr_classes]
        for idx, class_idx in enumerate(tqdm(idx_range,
                              file=sys.stdout,
                              bar_format='{l_bar}{' + f'bar:{ds_len * 2}' + '}{r_bar}{' + f'bar:-{ds_len * 2}b' + '}')):
            size_before_add = len(self.data)
            if use_preproc:
                path_to_img_dir = os.path.join(self.root_dir, 'by_class_preproc', hex(class_idx)[2:])
                self._add_samples_from_batches(path_to_img_dir, 'train' if self.train else 'test', idx, verify)
            else:
                newly_added = 0
                class_folder_name = hex(class_idx)[2:]

                if self.train:
                    for hsf_idx in range(0, 8):
                        hsf_name = 'hsf_' + str(hsf_idx)
                        path_to_img_dir = os.path.join(self.root_dir,
                                                       'by_class',
                                                       class_folder_name,
                                                       hsf_name)
                        if self._add_samples_from_dir(path_to_img_dir, idx, newly_added) == 1:
                            break
                else:
                    path_to_img_dir = os.path.join(self.root_dir,
                                                   'by_class',
                                                   class_folder_name,
                                                   'train_' + class_folder_name)
                    self._add_samples_from_dir(path_to_img_dir, idx, 0)
            self.classes[idx] = {'len': len(self.data) - size_before_add, 'chr': chr(class_idx)}
