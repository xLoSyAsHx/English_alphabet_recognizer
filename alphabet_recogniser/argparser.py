import argparse
import numpy as np

from alphabet_recogniser.datasets import NISTDB19Dataset


class ArgParser:
    __instance = None

    def __positive_int__(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
        return ivalue

    @staticmethod
    def get_instance():
        if ArgParser.__instance is None:
            ArgParser()
        return ArgParser.__instance

    def __init__(self):
        if ArgParser.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.parser = argparse.ArgumentParser(
                formatter_class=argparse.MetavarTypeHelpFormatter,
                description="Neural Network for English alphabet recognizer. "
                            "Example: python -m alphabet_recogniser.train -root-dir ./data -data-type low_letters -train-limit 1000 -test-limit 300 -e 30 --use-preprocessed-data -classes {a,b,f}"
            )

            # Required options
            self.parser.add_argument('-root-dir', type=str,  required=True,
                                     help="Path to data folder")
            self.parser.add_argument('-e', type=ArgParser.__positive_int__, required=True,
                                     help="Number of Epoches")
            self.parser.add_argument('-batch-size', type=ArgParser.__positive_int__, required=True,
                                     help="batch_size value for DataLoader")
            self.parser.add_argument('-data-type', type=str, choices=NISTDB19Dataset.folder_map.keys(), required=True,
                                     help=f"Specify data type to use. Available types: {NISTDB19Dataset.folder_map.keys()}")

            # Tensorboard options
            self.parser.add_argument('-t-images', type=ArgParser.__positive_int__,
                                     help="Specify number of samples from dataset to upload to tensorboard"
                                          "Must not exceed 100")
            self.parser.add_argument('-t-cm-granularity', type=ArgParser.__positive_int__,
                                     default=10,
                                     help="Specify how often to plot confusion matrix to upload to tensorboard")

            # Dataset settings
            self.parser.add_argument('-classes', type=str,
                                     help="Specify classes to use"
                                          "Example: -classes {a,b,c}")
            self.parser.add_argument('-train-limit', type=ArgParser.__positive_int__,
                                     help="Specify total num of train samples"
                                          "         (use '--limit-per-class' for per class limitation)")
            self.parser.add_argument('-test-limit', type=ArgParser.__positive_int__,
                                     help="Specify total num of samples"
                                          "         (use '--limit-per-class' for per class limitation)")
            self.parser.add_argument('--use-preprocessed-data', action='store_true',
                                     help="Set 'use_preproc=True'")
            self.parser.add_argument('--shuffle-train', action='store_true',
                                     help="Set 'shuffle=True' for train dataset")
            self.parser.add_argument('--shuffle-test', action='store_true',
                                     help="Set 'shuffle=True' for test dataset")

            # For uploading data from g-zipped archive
            self.parser.add_argument('-train-path', type=str,
                                     help="Path to zipped train dataset file")
            self.parser.add_argument('-test-path', type=str,
                                     help="Path to zipped test dataset file")

            ArgParser.__instance = self

    def parse_args(self):
        return self.parser.parse_args()

    def check_compatibility(self, args):
        if (args.train_path is not None or args.test_path is not None) and args.use_preprocessed_data:
            raise argparse.ArgumentError("'--use-preprocessed-data' not allowed with '-(train/test)-path'")

        if args.train_path is not None and args.test_path is None or \
                args.train_path is None and args.test_path is not None:
            raise argparse.ArgumentError("You must specify both '-train-path' and '-test-path'")

        if args.classes is not None and (args.classes[0] != '{' or args.classes[-1] != '}' or ' ' in args.classes):
            raise argparse.ArgumentError("Invalid format of argument for '-classes'"
                                         "Example: -classes {a,b,c}")

        if args.t_images is not None and args.t_images > 100:
            raise argparse.ArgumentError("Number of samples from dataset to upload must not exceed 100!")
