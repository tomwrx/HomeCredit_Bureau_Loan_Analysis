import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from collections import defaultdict

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)
 
    
class ValData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)
    
    
class BinaryClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(134, 670)
        self.layer_2 = nn.Linear(670, 670)
        self.layer_out = nn.Linear(670, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(670)
        self.batchnorm2 = nn.BatchNorm1d(670)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
    
def avg_precision(y_pred, y_test):

    y_pred_prob = torch.sigmoid(y_pred)
    y_pred_prob = y_pred_prob.cpu().detach().numpy()
    avg_precision = average_precision_score(y_test, y_pred_prob)

    return avg_precision


def torchclf_eval_test(
    y_test: np.ndarray, test_pred_list: np.ndarray, test_pred_labels: np.ndarray,
) -> pd.DataFrame:
    """Computes PyTorch binary classifier scikit learn metrics from the test set
    and stores them in a DataFrame"""

    scores = defaultdict(list)
    scores["Classifier"].append("PyTorch Binary Classifier")

    for metric in [
        balanced_accuracy_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
        roc_auc_score,
    ]:

        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        if score_name in ["Average precision", "Roc auc"]:

            scores[score_name].append(metric(y_test, test_pred_list))

        else:
            scores[score_name].append(metric(y_test, test_pred_labels))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df = score_df.round(3)

    return score_df