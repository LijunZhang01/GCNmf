"""
version 1.0
date 2021/02/04
"""

import argparse

from models import GCNmf
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask, Gendata
import json
import pandas as pd
import numpy as np
import torch
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=16, type=int, help='the number of hidden units')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
parser.add_argument('--epoch', default=10000, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose')

args = parser.parse_args()

def val_to_vec():
    data = pd.read_excel("./weixin_data/wx_data.xlsx")
    with open("./weixin_data/wx_data.json", 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    #Discrete_lis = ['轨道层次\nClass of Orbit', '轨道类型\nType of Orbit', '天线\nAntenna', '主体\nbody',  '太阳能电池板\nsolar panel', '按轨道高度\nby orbit altitude', '目的\nPurpose']
    Discrete_lis = ['Class of Orbit', 'Type of Orbit', 'Purpose']
    attr_lis = list(data.keys())
    #attr_lis.remove('名字')
    attr_lis.remove('Current Official Name of Satellite')
    attr = np.zeros((len(data), len(attr_lis)))
    #如何将data按照大小转化为标签label
    for i, k in enumerate(attr_lis):
        if k in Discrete_lis:
            for row in range(len(data)):
                if not pd.isnull(data.iloc[row, i+1]):  #字符串判空
                    attr[row, i] = data_dict[k].index(data.iloc[row, i+1])
                else:
                    attr[row, i] = -1
        else:
            for row in range(len(data)):
                if not pd.isnull(data.iloc[row, i+1]):
                    # attr[row, i] = val_num(data_dict[k], data.iloc[row, i+1])
                    attr[row, i] = data.iloc[row, i + 1]
                else:
                    attr[row, i] = -1
    return attr



def proc_data():
    data = torch.from_numpy(val_to_vec())  #获取各个属性对应的整型值 num_sat, num_attr
    data = data.long()
    data = data[data[:, -1]!=-1]


    mask = (data == -1)   #num_sat, num_attr
    feat_data = data[:, :-1]
    label = data[:, -1]
    mask = mask[:, :-1]
    return feat_data, label, mask





if __name__ == '__main__':
    # data = NodeClsData(args.dataset)
    feat_data, label, mask=proc_data()
    data = Gendata(feat_data, label)
    # mask = generate_mask(data.features, args.rate, args.type)
    # apply_mask(data.features, mask)
    model = GCNmf(data, nhid=args.nhid, dropout=args.dropout, n_components=args.ncomp)
    params = {
        'lr': args.lr,
        'weight_decay': args.wd,
        'epochs': args.epoch,
        'patience': args.patience,
        'early_stopping': True
    }
    trainer = NodeClsTrainer(data, model, params, niter=20, verbose=args.verbose)
    trainer.run()
