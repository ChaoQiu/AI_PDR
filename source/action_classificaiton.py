import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import os
import torch.nn.functional as F


class ActionRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ActionRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, bidirectional=False)
        self.f1 = nn.Sequential(nn.Linear(256, 128),
                                nn.Dropout(0.5),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, 10),
                                nn.softmax())

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.dropout(x, p=0.8)
        x = self.f1(x[:, -1, :])
        x = self.f2(x)
        return x


# def dataset():
# data = pd.read_csv(r'C:\Users\XYZ\Desktop\shuju_processed\shouchi\shouchi_hjh1.csv', usecols=[1, 2, 3], skiprows=0)
# # print(data.shape)
# data[0] = 0
# data.to_csv(r'C:\Users\XYZ\Desktop\2.csv', index=None)

# x_data = pd.read_csv(r'C:\Users\XYZ\Desktop\2.csv', usecols=[0, 1, 2])
# print(x_data)
# y_data = pd.read_csv(r'C:\Users\XYZ\Desktop\2.csv', usecols=[3])
# print(y_data)

class DealDataset(Dataset):
    def __init__(self):
        data = pd.read_csv(r'C:\Users\XYZ\Desktop\2.csv', usecols=[0, 1, 2, 3])
        self.x_data = pd.read_csv(r'C:\Users\XYZ\Desktop\2.csv', usecols=[0, 1, 2])
        self.y_data = pd.read_csv(r'C:\Users\XYZ\Desktop\2.csv', usecols=[3])
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dealDataset = DealDataset()

train_loader = DataLoader(dataset=dealDataset,
                          batch_size=32,
                          shuffle=True)
print(type(train_loader))
for i, data in enumerate(train_loader):
    print(data)




