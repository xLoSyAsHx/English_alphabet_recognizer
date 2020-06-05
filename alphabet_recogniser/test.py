import sys, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from alphabet_recogniser.argparser import ArgParser
from alphabet_recogniser.datasets import NISTDB19Dataset
from alphabet_recogniser.tensorboard_utils import log_conf_matrix, log_TPR_PPV_F1_bars, log_ROC_AUC
from alphabet_recogniser.utils import Config


class MLMetrics:
    def __init__(self, cm, pred_list, lbl_list, prob_list):
        self.cm = cm
        self.pred_list = pred_list
        self.lbl_list  = lbl_list
        self.prob_list = prob_list

        self.FP = FP = cm.sum(axis=0) - np.diag(cm)
        self.FN = FN = cm.sum(axis=1) - np.diag(cm)
        self.TP = TP = np.diag(cm)
        self.TN = TN = cm.sum() - (FP + FN + TP)

        eps = sys.float_info.min
        self.TPR = TP / (TP + FN + eps) # recall (true positive rate)
        self.TNR = TN / (TN + FP + eps)
        self.PPV = TP / (TP + FP + eps) # precision (positive prediction value)
        self.NPV = TN / (TN + FN + eps)
        self.FPR = FP / (FP + TN + eps)
        self.FNR = FN / (TP + FN + eps)
        self.FDR = FP / (TP + FP + eps)
        self.F1  = 2 * (self.PPV * self.TPR) / (self.PPV + self.TPR + eps)
        self.ACC = (TP + TN) / (TP + FP + FN + TN + eps)


def eval(net, test_loader, epoch, log_loss=True):
    config = Config.get_instance()
    with torch.no_grad():
        pred_list = torch.zeros(0, dtype=torch.long,  device='cpu')
        lbl_list  = torch.zeros(0, dtype=torch.long,  device='cpu')
        prob_list = torch.zeros(0, dtype=torch.float, device='cpu')
        for i, data in enumerate(test_loader):
            images, labels = data[0].to(config.device), data[1].to(config.device)

            outputs = net(images)
            if log_loss:
                config.writer.add_scalar('Loss/test', config.criterion(outputs, labels).item(), epoch)

            p_values, p_indexes = torch.max(outputs.data, 1)
            pred_list = torch.cat([pred_list, p_indexes.view(-1).cpu()])
            lbl_list  = torch.cat([lbl_list, labels.view(-1).cpu()])
            prob_list = torch.cat([prob_list, p_values.view(-1).cpu()])

        cm = confusion_matrix(lbl_list, pred_list)
        return MLMetrics(cm, pred_list, lbl_list, prob_list)


def eval_cached(net, test_loader, epoch, log_loss=False):
    if hasattr(eval_cached, 'last_calculated_epoch') and eval_cached.last_calculated_epoch == epoch:
        return eval_cached.metrics

    eval_cached.last_calculated_epoch = epoch
    eval_cached.metrics = eval(net, test_loader, epoch, log_loss)

    return eval_cached.metrics


def main():
    parser = argparse.ArgumentParser(description="Neural Network test for English alphabet recognizer")
    parser.add_argument('-root-dir', type=str,                  required=True, help="Path to data folder")
    parser.add_argument('-model', type=ArgParser.__sys_path__,  required=True, help="Specify path to model")
    parser.add_argument('-n', type=ArgParser.__positive_int__,  required=True, help="Specify num of test samples")
    parser.add_argument('-data-type', type=str, choices=NISTDB19Dataset.folder_map.keys(), required=True,
                        help=f"Specify data type to use. Available types: {NISTDB19Dataset.folder_map.keys()}")
    parser.add_argument('-classes', type=ArgParser.__char_unique_array__,      help="Specify classes to use"
                                                                                    "Example: -classes {a,b,c}")

    args = parser.parse_args()
    config = Config.get_instance()

    NISTDB19Dataset.download_and_preprocess(root_dir=args.root_dir, data_type=args.data_type, str_classes=args.classes)
    test_set = NISTDB19Dataset(root_dir=args.root_dir, data_type=args.data_type, train=False, download=True,
                               str_classes=args.classes, use_preproc=True, size_limit=args.n,
                               test_transform=transforms.Compose(
                                                [transforms.CenterCrop(96),
                                                 transforms.Grayscale(num_output_channels=1),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))]))
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Eval and show statistics
    net = torch.load(args.model)  # EngAlphabetRecognizer96
    metrics = eval(net, test_loader, None, log_loss=False)

    print(f'Accuracy of the network with {len(test_set.classes)} classes ({args.n} el per class): {100 * np.mean(metrics.TPR):3.2f}%')
    for idx, target in enumerate(test_set.classes):
        print(f"Recall of '{test_set.classes[idx]['chr']}': {100 * metrics.TPR[target]:3.2f}%")

    classes = [test_set.classes[key]['chr'] for key in test_set.classes]
    log_conf_matrix(metrics, classes, None)
    log_TPR_PPV_F1_bars(metrics, classes, None)
    log_ROC_AUC(metrics, classes, None)
    plt.show()


if __name__ == "__main__":
    main()
