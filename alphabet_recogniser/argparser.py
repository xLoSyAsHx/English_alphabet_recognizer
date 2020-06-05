import os
import argparse

from alphabet_recogniser.datasets import NISTDB19Dataset


class ArgParser:
    __args__ = None

    def __positive_int__(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
        return ivalue

    def __sys_path__(path):
        if path is not None and not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"Path '{path}' doesn't exist")
        return path

    def __char_unique_array__(str):
        if str is not None and (str[0] != '{' or str[-1] != '}' or ' ' in str):
            raise argparse.ArgumentError("Invalid format of argument for '-classes'"
                                         "Example: -classes {a,b,c}")
        return str

    @staticmethod
    def get_args():
        if ArgParser.__args__ is None:
            ArgParser()
        return ArgParser.__args__

    def __init__(self):
        if ArgParser.__args__ is not None:
            raise Exception("This class is a singleton!")
        else:
            # Load args from config file
            parser = argparse.ArgumentParser(description="Neural Network for English alphabet recognizer")
            parser.add_argument('-config', type=ArgParser.__sys_path__, help="Specify path to config.cfg file")
            path_to_cfg = parser.parse_args().config

            args_to_process = []
            with open(path_to_cfg, 'r') as cfg_file:
                for line in cfg_file.read().splitlines():
                    arg = line.lstrip().split(' ') if len(line) != 0 else []
                    if len(arg) == 0 or arg[0] == '#':
                        continue
                    elif len(arg) == 2 or (len(arg) == 1 and arg[:2] != '--'):
                        args_to_process.extend(arg)
                    else:
                        raise argparse.ArgumentError(f"Invalid format of '{path_to_cfg}' file"
                                                     f"Example:"
                                                     f"    'argument_key1 argument_value1'"
                                                     f"    'argument_key2 argument_value2'")

            # Process args from config
            parser = argparse.ArgumentParser(
                formatter_class=argparse.MetavarTypeHelpFormatter,
                description="Example: python -m alphabet_recogniser.train --config path/to/config.cfg"
            )

            # Required options
            parser.add_argument('-root-dir', type=str, required=True,
                                help="Path to data folder")
            parser.add_argument('-e', type=ArgParser.__positive_int__, required=True,
                                help="Number of Epoches")
            parser.add_argument('-batch-size', type=ArgParser.__positive_int__, required=True,
                                help="batch_size value for DataLoader")
            parser.add_argument('-data-type', type=str, choices=NISTDB19Dataset.folder_map.keys(), required=True,
                                help=f"Specify data type to use. Available types: {NISTDB19Dataset.folder_map.keys()}")

            # Model options
            parser.add_argument('-m-load-path', type=ArgParser.__sys_path__,
                                help="Specify path to model for load")
            parser.add_argument('-m-save-path', type=ArgParser.__sys_path__,
                                help="Specify path to folder for save trained model")
            parser.add_argument('-m-save-period', type=ArgParser.__positive_int__,
                                help="Specify how often save trained model")

            # Tensorboard options
            parser.add_argument('-t-logdir', type=ArgParser.__sys_path__,
                                default='./runs/',
                                help="Specify path to tensorboard logs"
                                     "Example: -t-logdir ./runs/")
            parser.add_argument('-t-images', type=ArgParser.__positive_int__,
                                choices=range(0, 101), metavar="[0-100]",
                                help="Specify number of samples (0-100) from dataset to upload to tensorboard")
            parser.add_argument('-t-cm-freq', type=ArgParser.__positive_int__,
                                default=10,
                                help="Specify how often to plot confusion matrix to upload to tensorboard")
            parser.add_argument('-t-precision-bar-freq', type=ArgParser.__positive_int__,
                                default=10,
                                help="Specify how often to plot precision bar to upload to tensorboard")
            parser.add_argument('-t-roc-auc-freq', type=ArgParser.__positive_int__,
                                default=10,
                                help="Specify how often to plot ROC curves to upload to tensorboard")

            # Dataset settings
            parser.add_argument('-classes', type=ArgParser.__char_unique_array__,
                                help="Specify classes to use"
                                     "Example: -classes {a,b,c}")
            parser.add_argument('-train-limit', type=ArgParser.__positive_int__,
                                help="Specify total num of train samples"
                                     "         (use '--limit-per-class' for per class limitation)")
            parser.add_argument('-test-limit', type=ArgParser.__positive_int__,
                                help="Specify total num of samples"
                                     "         (use '--limit-per-class' for per class limitation)")
            parser.add_argument('--use-preprocessed-data', action='store_true',
                                help="Set 'use_preproc=True'")
            parser.add_argument('--shuffle-train', action='store_true',
                                help="Set 'shuffle=True' for train dataset")
            parser.add_argument('--shuffle-test', action='store_true',
                                help="Set 'shuffle=True' for test dataset")

            # For load/save data to/from lzima-zipped archive
            parser.add_argument('-train-save-path', type=ArgParser.__sys_path__,
                                help="Path to save lzima-zipped train dataset file"
                                     "Will use NISTDB19Dataset.save_to_file(dataset, path)")
            parser.add_argument('-test-save-path', type=ArgParser.__sys_path__,
                                help="Path to save lzima-zipped test dataset file"
                                     "Will use NISTDB19Dataset.save_to_file(dataset, path)")
            parser.add_argument('-train-load-path', type=ArgParser.__sys_path__,
                                help="Path to lzima-zipped train dataset file"
                                     "Will use NISTDB19Dataset.load_from_file(path)")
            parser.add_argument('-test-load-path', type=ArgParser.__sys_path__,
                                help="Path to lzima-zipped test dataset file"
                                     "Will use NISTDB19Dataset.load_from_file(path)")

            ArgParser.__args__ = parser.parse_args(args_to_process)
