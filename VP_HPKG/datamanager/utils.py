import argparse
import multiprocessing
from collections import defaultdict
from operator import index
from random import random
from tkinter import ON

import numpy as np
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#预处理数据获取
from data_test import *

#参数设置
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')
    
    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=8,
                        help='Number of node dimensions. Default is 200.')

    parser.add_argument('--edge-dim', type=int, default=4,
                        help='Number of edge embedding dimensions. Default is 10.')
    
    parser.add_argument('--att-dim', type=int, default=4,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--window-size', type=int, default=2,
                        help='Context size for optimization. Default is 5.')
    
    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')
    
    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.') #邻居采样

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.') #早停机制，在过拟合之前停止训练

    return parser.parse_args()

#加载训练集数据
def load_train_data():
    print('loading training data!!!')
    dict_edge = get_edge()
    all_nodes = list()
    edge_data_by_type = dict()
    for i in range(len(dict_edge)):
        if dict_edge[i][1] not in edge_data_by_type:
            edge_data_by_type[dict_edge[i][1]] = list()
        x , y = dict_edge[i][0] , dict_edge[i][2]
        edge_data_by_type[dict_edge[i][1]].append((x , y))
        all_nodes.append(x)
        all_nodes.append(y)
    all_nodes = list(set(all_nodes)) #去除重复的节点 
    #print(all_nodes)
    #print(edge_data_by_type)
    return all_nodes, edge_data_by_type

#存储每个节点的SDC率 作为y 前面6个属性作为x 
def get_SDC(features):
    feature_y = {}  #存储每个节点的SDC率
    keys = list(features.keys())
    #print(keys)
    for i in range(len(keys)):
        feature_y[keys[i]] = features[keys[i]][-1]
        features[keys[i]].pop()#删除最后一个属性SDC率

    return features, feature_y
#获得KG的邻接矩阵
def get_adj():
    all_nodes, edge_data_by_type = load_train_data()
    num_node = len(all_nodes)#节点总数
    keys = list(edge_data_by_type.keys())
    num_type_edge = len(list(keys)) #边的类型总数

    #创建邻接矩阵
    #print(edge_data_by_type)
    adj = []
    for i in range(num_type_edge):
        r_adj = np.zeros((num_node,num_node), dtype = float) #生成全0的nxn的二维数组
        for j in range(len(edge_data_by_type[keys[i]])):
            r_adj[int(edge_data_by_type[keys[i]][j][0])][int(edge_data_by_type[keys[i]][j][1])] = 1
        print(r_adj)
        break

    return 
def load_edge_data():

    return get_edge()

#加载节点特征, 并将SDC作为label
def load_feature_data():
    print("load node features!!!")
    features = get_features()
    #get_SDC(features)

    #进行编码 将字符串型的属性转为数值型属性
    all_feature = []
    for key , value in features.items():
        all_feature.append(np.array(value))
    all_feature = np.array(all_feature)
    lc = LabelEncoder()
    #print(all_feature[:,4]) #取二维numpy数组的某一列
    for i in range(4,len(all_feature[0])-1):
        all_feature[:,i] = lc.fit_transform(all_feature[:,i])

    labels = all_feature[:, -1]
    #目前SDC率正在故障注入，后期还需要进行处理，先随机生成以便试验

    return all_feature[:, :-1] , labels
#加载基本块的信息 并进行预处理
def load_BB_info():
    dict_BB , edge = get_BB_info()
    nodes = list(dict_BB.keys()) #基本块标号
    #print(nodes)
    #print(list(dict_BB.values()))
    features = [] #将节点编号与特征序列在表中的位置对应
    for i in range(len(nodes)):
        features.append(dict_BB[str(i)])
    #print(features)
    return nodes, features, edge

if __name__ == '__main__':
    nodes, edge = load_train_data()
    #features = load_feature_data()
    #features = np.asarray(features, int)
    #print(nodes, edge)
    load_BB_info()
