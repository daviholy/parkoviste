import csv

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def count_roc_auc(n_classes, output, labels, plot=False):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(labels[:, i]), np.array(output[:, i]))
        roc_auc[i] = auc(fpr[i], tpr[i])

        if plot:
            plt.figure()
            lw = 2
            plt.plot(
                fpr[i],
                tpr[i],
                color="darkorange",
                lw=lw,
                label="ROC curve (area = %0.2f)" % roc_auc[i],
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Receiver operating characteristic example {i}")
            plt.legend(loc="lower right")
            plt.show()

    return fpr, tpr, roc_auc


def draw_loss(csv_file_path):

    train_loss = []
    test_loss = []
    epochs = []
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            epochs.append(int(row[0]))
            train_loss.append(float(row[1]))
            test_loss.append(float(row[2]))

    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")

    plt.title('Loss during training')
    plt.xlabel('epochs')
    plt.ylabel('average loss per batch')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # TODO: filename as parameter
    draw_loss('../../models/model_006/model_info.csv')
