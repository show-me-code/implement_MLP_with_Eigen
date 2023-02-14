import csv
from builtins import Exception, print
from calendar import prcal
import torch
import numpy as np
import math
import time
import json
import os
from easydict import EasyDict as edict
import torch.profiler
import os
import pandas as pd
from collections import OrderedDict
pd.set_option('display.max_rows', None)#显示全部行
pd.set_option('display.max_columns', None)#显示全部列
pd.set_option('display.width',None)
np.set_printoptions(threshold=np.inf)

class MyGELU(torch.nn.Module):
    def __init__(self):
        super(MyGELU, self).__init__()
        self.torch_PI = 3.1415926536

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / self.torch_PI) * (x + 0.044715 * torch.pow(x, 3))))

def json2Parser(json_path):
    """load json and return parser-like object"""
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        neurons = layers
        self.depth = len(neurons) - 1
        self.actfun = MyGELU()
        self.layers = []
        for i in range(self.depth - 1):
            self.layers.append(torch.nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(self.actfun)
        self.layers.append(torch.nn.Linear(neurons[-2], neurons[-1]))  # last layer
        self.fc = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.fc(x)
        return x
setting0 = json2Parser('settingsdrm19_0.json')
setting1 = json2Parser('settingsdrm19_1.json')
setting2 = json2Parser('settingsdrm19_2.json')

lamda = setting0.power_transform
delta_t = setting0.delta_t
dim = setting0.dim
layers = setting0.layers

def save_bin(data, bin_file, dtype="double"):
    """
    C++int对应Python np.intc
    C++float对应Python np.single
    C++double对应Python np.double
    :param data:
    :param bin_file:
    :param dtype:
    :return:
    """
    data = data.astype(np.double)
    data.astype(dtype).tofile(bin_file)


def load_bin(bin_file, shape=None, dtype="double"):
    """
    :param bin_file:
    :param dtype:
    :return:
    """
    data = np.fromfile(bin_file, dtype=dtype)
    if shape:
        data = np.reshape(data, shape)
    return data

if __name__ == '__main__':
    model0 = Net()
    model0.to('cpu')
    check_point = torch.load('./modeldrm19_2.pt')
    model0.load_state_dict(check_point)

    #model0 = torch.load("./model5000.pt")
    #with open('text.txt', 'a') as file:
    #    print(model0, file=file)
    #for key, val in model0.items():
    #    print(key, val)
    #print(model0.state_dict())
    model_dict  = dict(model0.state_dict())
    new_dict = {}
    for k, v in model_dict.items():
        v = v.numpy()
        print(np.shape(v))
        #new_dict[k] = v
        with open('layer_model2.txt', 'a') as f:
            f.writelines(k + str(np.shape(v)) + '\n')
        save_bin(v, k+'_model2.bin')    #print(new_dict)
        #print(v)
    data2 = load_bin('fc.0.bias_model2.bin', 3200)
    df = pd.DataFrame.from_dict(model_dict, orient='index')
    #print(df)
    df.to_csv('model2.csv')