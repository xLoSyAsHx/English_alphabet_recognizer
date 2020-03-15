import argparse

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
            self.parser = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)

            self.parser.add_argument('-train-path', type=str,
                                     help="Path to zipped train dataset file")
            self.parser.add_argument('-test-path', type=str,
                                     help="Path to zipped test dataset file")
            self.parser.add_argument('--shuffle-train', type=bool, default=False,
                                     help="Set 'shuffle=True' for train dataset")
            self.parser.add_argument('--shuffle-test', type=bool, default=False,
                                     help="Set 'shuffle=True' for test dataset")
            self.parser.add_argument('-data-type', type=str, choices=NISTDB19Dataset.folder_map.keys(),
                                     help=f"Specify data type to use. Available types: {NISTDB19Dataset.folder_map.keys()}")
            self.parser.add_argument('-e', type=ArgParser.__positive_int__, required=True,
                                     help="Number of Epoches")
            self.parser.add_argument('-train-limit', type=ArgParser.__positive_int__,
                                     help="Specify total num of train samples"
                                          "         (use '--limit-per-class' for per class limitation)")
            self.parser.add_argument('-test-limit', type=ArgParser.__positive_int__,
                                     help="Specify total num of test samples"
                                          "         (use '--limit-per-class' for per class limitation)")
            self.parser.add_argument('--use-preprocessed-data', action='store_true',
                                     help="Set 'use_preproc=True'")

            ArgParser.__instance = self

    def parse_args(self):
        return self.parser.parse_args()

    def check_compatibility(self, args):
        if (args.train_path is not None or args.test_path is not None) and args.use_preprocessed_data:
            raise argparse.ArgumentError("'--use-preprocessed-data' not allowed with '-(train/test)-path'")

        if args.train_path is not None and args.test_path is None or \
                args.train_path is None and args.test_path is not None:
            raise argparse.ArgumentError("You must specify both '-train-path' and '-test-path'")
