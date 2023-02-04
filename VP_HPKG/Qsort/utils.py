import argparse
import multiprocessing
from collections import defaultdict
from operator import index
from random import random
from tkinter import ON

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from data_test import *

#Parameter setting
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

    return parser.parse_args()

#Load training set data and return instruction entity set and edge set
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
    all_nodes = list(set(all_nodes)) #Remove duplicate nodes 
    return all_nodes, edge_data_by_type

#Get Edge Set
def load_edge_data():
    return get_edge()

#Load node attributes and use error rate as label
def load_feature_data():
    print("load node features!!!")
    features = get_features()
    all_feature = []
    for key , value in features.items():
        all_feature.append(np.array(value))
    all_feature = np.array(all_feature)
    lc = LabelEncoder() #Encoder
    for i in range(4,len(all_feature[0])-1):
        all_feature[:,i] = lc.fit_transform(all_feature[:,i])

    labels = all_feature[:, -1] #label
    return all_feature[:, :-1] , labels
#Load basic block information and preprocess
def load_BB_info():
    dict_BB , edge = get_BB_info()
    nodes = list(dict_BB.keys()) #Basic block label
    features = [] #Corresponds the node number to the position of the attribute sequence in the table
    for i in range(len(nodes)):
        features.append(dict_BB[str(i)])
    return nodes, features, edge

if __name__ == '__main__':
    nodes, edge = load_train_data()
    load_BB_info()
