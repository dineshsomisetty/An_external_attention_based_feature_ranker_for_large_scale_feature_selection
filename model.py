import torch.nn as nn
import torch


class FeatureSelectionMLP(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()
        self.selection_rate = nn.Parameter(
            # (torch.rand([1, n_features]) - 0.56) / 4
            torch.ones([1, n_features])
            # torch.rand([1, n_features])
        )

    def get_selection_rate(self):
        return torch.sigmoid(self.selection_rate)

    def forward(self, x):
        return x * torch.sigmoid(self.selection_rate)


class Classifier_pointConv(nn.Module):
    def __init__(self, n_features, n_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=4 * n_features, kernel_size=(1, 1),
                               groups=n_features)
        self.conv2 = nn.Conv1d(in_channels=4 * n_features, out_channels=4 * n_features, kernel_size=(1, 1),
                               groups=n_features)
        self.conv3 = nn.Conv1d(in_channels=4 * n_features, out_channels=n_classes, kernel_size=(1, 1),
                               groups=n_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(nn.ReLU(x))
        x = self.conv3(nn.ReLU(x))
        return x


# class Classifier_mconv(nn.Module):
#     def __init__(self,):

class Classifier(nn.Module):
    def __init__(self, n_features, n_classes) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features, int(n_features * 4)),  # used to 8
            nn.ReLU(),
            nn.Linear(int(n_features * 4), int(n_features * 2)),
            nn.ReLU(),
            nn.Linear(int(n_features * 2), n_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class Model(nn.Module):
    def __init__(self, n_features, n_classes) -> None:
        super().__init__()
        self.classifier = Classifier(n_features, n_classes)
        self.feature_selection_MLP = FeatureSelectionMLP(n_features)

    def forward(self, x):
        return self.classifier(
            self.feature_selection_MLP(x)
        )


def select_feature(origin_f, select_rate, threshold):  # 使用原始特性和注意力图，生成重要特征
    (a, l) = select_rate.shape
    # print(l)
    flag = False
    x = None
    for i in range(l):
        # print(f'{i}, {float(select_rate[0, i])}')
        if float(select_rate[0, i]) != 0:  # > threshold
            if not flag:
                x = origin_f[:, i]
                flag = True
            else:
                x = torch.vstack([x, origin_f[:, i]])
    return x.permute(1, 0)


def sorted_idx(select_rate):  # 从大到小返回在原序列中的位置
    a = select_rate[0, :]
    idx = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
    return idx


def select_idx_F(origin_f, idx, num):  # 返回rate最大的前num个特征
    x = origin_f[:, idx[0]]
    for i in range(1, num):
        x = torch.vstack([x, origin_f[:, idx[i]]])
    return x.permute(1, 0)


def add_one_features(origin, inp, idx, n):
    x = inp.permute(1, 0)
    x = torch.vstack([x, origin[:, idx[n-1]]])
    return x.permute(1, 0)


def add_more_features(origin, inp, idx, n_max, n):
    x = inp.permute(1, 0)
    for i in range(n, n_max):
        x = torch.vstack([x, origin[:, idx[i]]])
    return x.permute(1, 0)

def log(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    logfile = path
    fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)  # 输出到console的log等级的开关

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix
