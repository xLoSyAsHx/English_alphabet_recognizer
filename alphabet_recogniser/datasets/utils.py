from torch.utils.data import DataLoader

from alphabet_recogniser.datasets import NISTDB19Dataset
from alphabet_recogniser.utils import Config


def NISTDB19Dataset_data_loaders(force_shuffle_test=True):
    self = NISTDB19Dataset_data_loaders
    C = Config.get_instance()
    if hasattr(self, 'train'):
        return self.train, self.test

    if C.args.use_preprocessed_data:
        NISTDB19Dataset.download_and_preprocess(root_dir=C.args.root_dir, data_type=C.args.data_type,
                                                str_classes=C.args.classes)

    if C.args.train_load_path is None:
        train_set = NISTDB19Dataset(root_dir=C.args.root_dir, data_type=C.args.data_type, train=True, download=True,
                                    str_classes=C.args.classes, use_preproc=C.args.use_preprocessed_data,
                                    train_transform=C.train_transform, test_transform=C.test_transform,
                                    size_limit=C.args.train_limit)
        if C.args.train_save_path is not None:
            NISTDB19Dataset.save_to_file(train_set, C.args.train_save_path)
    else:
        train_set = NISTDB19Dataset.load_from_file(C.args.train_load_path)
    self.train = DataLoader(train_set, batch_size=C.args.batch_size,
                            shuffle=C.args.shuffle_train, num_workers=0)

    if C.args.test_load_path is None:
        test_set = NISTDB19Dataset(root_dir=C.args.root_dir, data_type=C.args.data_type, train=False, download=True,
                                   str_classes=C.args.classes, use_preproc=C.args.use_preprocessed_data,
                                   train_transform=C.train_transform, test_transform=C.test_transform,
                                   size_limit=C.args.test_limit)
        if C.args.test_save_path is not None:
            NISTDB19Dataset.save_to_file(test_set, C.args.test_save_path)
    else:
        test_set = NISTDB19Dataset.load_from_file(C.args.test_load_path)
    self.test = DataLoader(test_set, batch_size=C.args.batch_size,
                           shuffle=C.args.shuffle_test if force_shuffle_test is False else True, num_workers=0)

    C.classes = train_set.classes
    return self.train, self.test
