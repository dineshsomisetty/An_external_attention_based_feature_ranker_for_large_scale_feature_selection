import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from model import Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from model import select_idx_F
from model import sorted_idx
from model import add_more_features
from model import add_one_features
from model import log
import warnings

warnings.filterwarnings("ignore")
clfl = [KNeighborsClassifier(n_neighbors=3), svm.LinearSVC(dual=False), DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=0)]
path = "dataset.csv"
lr = 0.01
device = torch.device('cuda:0')
epoch = 200
batch_size = 100
validation_split = .3
threshold = 0.6


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# data = pd.read_csv(path, header=None)
# d = data.iloc[1:, :]
# data = np.array(d)
data = np.loadtxt(path)
data, label = data[:, :-1], data[:, -1]  # 默认最后一列为label
(a, n_feature) = data.shape
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)
num_label = max(label) + 1
# num_label = max(label)
# label = label - 1
# for i in range(a):  # gisette madelon
#     if label[i] == -1:
#         label[i] = 0

data1, label1 = data, label
data, label = torch.from_numpy(data).float(), torch.from_numpy(label).long()
dataset = Data.TensorDataset(data, label)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = Data.Subset(dataset, train_indices)
valid_sampler = Data.Subset(dataset, val_indices)
train_loader = Data.DataLoader(train_sampler, batch_size=batch_size,
                               )
validation_loader = Data.DataLoader(valid_sampler, batch_size=batch_size,
                                    )
logpath = './log.txt'
logger = log(logpath)

# rd = []
for random_seed in rd:  # rd is 30 random numbers type list []
    setup_seed(random_seed)
    model = Model(n_feature, int(num_label)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epoch, 1, eta_min=1e-3)
    loss_fun = nn.CrossEntropyLoss()
    for i in range(epoch):
        acc = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fun(output, targets)
            loss += 1e-2 / torch.sum(
                (model.feature_selection_MLP.get_selection_rate() - 0.5) ** 2)
            print(loss)
            loss.backward()
            optimizer.step()  #
            predicts = output.argmax(dim=1)
            acc += torch.eq(predicts, targets).sum().float().item()
        scheduler.step()
        # print(optimizer.param_groups[0]['lr'], acc / data.size(0))
        logger.info(
            f'EPOCH: {i} train: lr: {optimizer.param_groups[0]["lr"]} " " acc rate:{acc / train_sampler.__len__()}  dirty rate =  split {validation_split}')
        if i % 10 == 0:
            test_acc = 0
            with torch.no_grad():
                for inputs, targets in validation_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    # optimizer.zero_grad()
                    output = model(inputs)
                    predicts = output.argmax(dim=1)
                    test_acc += torch.eq(predicts, targets).sum().float().item()
                logger.info(
                    f'EPOCH:{i}test: lr :{optimizer.param_groups[0]["lr"]}  " " acc rate:{test_acc / valid_sampler.__len__()}  split {validation_split}')

    select_rate = model.feature_selection_MLP.get_selection_rate()
    torch.save(select_rate, 'select_rate.pt')
    count = -1
    record = []
    for clf in clfl:
        count = count + 1
        # print(count)
        name = l[count]  # l is clf's name ['knn,svm...']
        lit = [i for i in np.arange(0.02, 1.01, 0.02)]
        lis = [int(i * n_feature) for i in lit]
        idx = sorted_idx(select_rate)
        y = []
        plt.figure(dpi=80)
        x = lit
        ct = False
        for i in lis:
            if not ct:
                selected_f = select_idx_F(torch.from_numpy(data1), idx, int(i))
                ct = True
            else:
                selected_f = add_more_features(torch.from_numpy(data1), selected_f, idx, int(i), int(j))
            j = i
            print(f'{random_seed}  {int(i)}')

            # print(selected_f.shape)
            data_train, data_test, target_train, target_test = train_test_split(selected_f, label1,
                                                                                test_size=validation_split,
                                                                                random_state=42)
            # print(data_train.shape)
            # score = cross_val_score(clf,data_train,target_train,cv=3)
            clf.fit(data_train, target_train)
            score = clf.score(data_test, target_test)
            logger.info(f'Feature_shape: {selected_f.shape}  Score: {score}')
            y.append(score)
            # print(score)
        m = max(y)
        i = y.index(m)
        prop = (i + 1) * 0.02
        print(prop)
        print(m)
        features = round(n_feature * prop) - 25
        plt.scatter(x, y, label='Subsets', marker='*')
        plt.xlabel('Proportion of features')  # Number Proportion
        plt.ylabel('Accuracy')
        plt.title('Accuracy Change with Proportion of Features')
        plt.legend(bbox_to_anchor=(1, 1),
                   loc="upper right",
                   ncol=1,
                   mode="None",
                   borderaxespad=0,
                   shadow=False,
                   fancybox=True)
        plt.savefig(f'./{str(random_seed)}/png/{name}_subset_prop.png')
        plt.show()
        np.save(f'./{str(random_seed)}/prop/{name}y.npy', np.array(y))  # subg_1的精度
        low = features if features > 1 else 2
        high = low + 50
        if high > n_feature:
            high = n_feature
            low = high - 50
        lit_n = [i for i in range(low, high)]
        y = []
        plt.figure(dpi=80)
        x = lit_n
        ct_1 = False
        for i in lit_n:
            if not ct_1:
                selected_f = select_idx_F(torch.from_numpy(data1), idx, int(i))
                ct_1 = True
            else:
                selected_f = add_one_features(torch.from_numpy(data1), selected_f, idx, int(i))
            print(f'{random_seed}  {int(i)}')
            # print(selected_f.shape)
            data_train, data_test, target_train, target_test = train_test_split(selected_f, label1,
                                                                                test_size=validation_split,
                                                                                random_state=42)
            clf.fit(data_train, target_train)
            score = clf.score(data_test, target_test)
            logger.info(f'Feature_shape: {selected_f.shape}  Score: {score}')
            y.append(score)
            # print(score)
        m = max(y)
        Y = y.index(max(y)) + low
        print(Y)
        print(m)
        record.append(m)
        record.append(Y)
        plt.scatter(x, y, label='Subsets', marker='*')
        plt.scatter(Y, m, marker='*')
        show_max = '[' + str(Y) + ' ' + str(round(max(y), 4)) + ']'
        plt.annotate(show_max, xytext=(Y, m + 0.0001), xy=(Y, m))
        plt.xlabel('Number of features')  # Number Proportion
        plt.ylabel('Accuracy')
        plt.title('Accuracy Change with Number of Features')
        plt.legend(bbox_to_anchor=(1, 1),
                   loc="upper right",
                   ncol=1,
                   mode="None",
                   borderaxespad=0,
                   shadow=False,
                   fancybox=True)
        plt.savefig(f'./{str(random_seed)}/png/{name}subset.png')
        plt.show()
        np.save(f'./{str(random_seed)}/y/{name}y.npy', np.array(y))  # subg_2的精度
    np.save(f'./{str(random_seed)}/record.npy', record)
