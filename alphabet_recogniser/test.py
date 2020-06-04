import sys
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


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


def eval(G, net, test_loader, epoch, log_loss=True):
    with torch.no_grad():
        pred_list = torch.zeros(0, dtype=torch.long,  device='cpu')
        lbl_list  = torch.zeros(0, dtype=torch.long,  device='cpu')
        prob_list = torch.zeros(0, dtype=torch.float, device='cpu')
        for i, data in enumerate(test_loader):
            images, labels = data[0].to(G.device), data[1].to(G.device)

            outputs = net(images)
            if log_loss:
                G.writer.add_scalar('Loss/test', G.criterion(outputs, labels).item(), epoch)

            p_values, p_indexes = torch.max(outputs.data, 1)
            pred_list = torch.cat([pred_list, p_indexes.view(-1).cpu()])
            lbl_list  = torch.cat([lbl_list, labels.view(-1).cpu()])
            prob_list = torch.cat([prob_list, p_values.view(-1).cpu()])

        cm = confusion_matrix(lbl_list, pred_list)
        return MLMetrics(cm, pred_list, lbl_list, prob_list)


def eval_cached(G, net, test_loader, epoch, log_loss=False):
    if hasattr(eval_cached, 'last_calculated_epoch') and eval_cached.last_calculated_epoch == epoch:
        return eval_cached.metrics

    eval_cached.last_calculated_epoch = epoch
    eval_cached.metrics = eval(G, net, test_loader, epoch, log_loss)

    return eval_cached.metrics


def main():
    pass


if __name__ == "__main__":
    main()
