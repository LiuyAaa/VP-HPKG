from ast import Param
from email.contentmanager import raw_data_manager
from locale import normalize
import math
from operator import mod
from telnetlib import GA
from tkinter.tix import ListNoteBook
from turtle import color, forward, hideturtle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.parameter import Parameter
import dgl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import networkx as nx
from sklearn.manifold import TSNE
import seaborn as sns
import time

import dgl.nn as dglnn
import os
from utils import *
random_seed = 1388
np.random.seed(random_seed)
torch.manual_seed(random_seed)
path1 = os.path.dirname(os.path.abspath(__file__))

t0 = 0
t1 = 0

#Edge type
edge_type = {1:'control', 2:'data', 3:'fun_call', 4:'load_store', 5:'jump'}

#Standardization
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals)/ranges
    return np.around(normData,4)

#Map node attributes to nodes of heterogeneous graph
def to_hetero_feat(h, type, name):
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[torch.where(type == index)]
    return h_dict

import dgl.function as fn
attention_score = [0]*4 #Edge weight of the model

class HeteroRGCN(nn.Module):
    def __init__(self, BB_insize, BB_hidden, in_size, hidden_size, h_size,  out_size, BB_ins):
        super(HeteroRGCN, self).__init__()

        self.BBins = BB_ins #The instructions contained in the basic block are stored in numbered order
        # Basic block layer, GCN
        self.BBconv = dglnn.GraphConv(BB_insize, BB_hidden, norm= 'both', weight = True, bias = True, allow_zero_in_degree = True )
       
        # Instruction layer, Heterogeneous Graph Transformer, Get global correlation between nodes
        self.Hgtconv = dglnn.HGTConv(in_size+BB_hidden, hidden_size, 2, 1, 5) #Modified the source code of dgl HGTConv, Add edge weights to the output

        #output layer
        self.dense = nn.Linear(2*hidden_size, out_size)

        nn.init.uniform_(self.dense.weight, a=-0.1, b =0.1)
        nn.init.constant_(self.dense.bias, 0.1)

    def forward(self, G, BB_G, f, BB_f):
        #G : Instruction Heterogeneous Graph
        #BB_G: Basic block subgraph
        #f: The attributes of instruction
        #BB_f: he attributes of basic block

        #Basic block embedding
        res = self.BBconv(BB_G, BB_f)
        feature = f['node'] #
        ins_number = len(feature)
        temp = [[]]*ins_number

        #Instruction Attribute Aggregation Basic Block Embedding
        for i in range(len(self.BBins)):
            temp[i] =res[int(self.BBins[i])].detach().numpy()
        temp = torch.FloatTensor(np.array(temp))
        temp = torch.cat([feature, temp], dim = 1)

        #Instruction embedding
        with G.local_scope():
            G.ndata['h'] = temp
            g = dgl.to_homogeneous(G, ndata='h')
            h = g.ndata['h']
            h, g1 = self.Hgtconv(g, h, g.ndata['_TYPE'], g.edata['_TYPE'], presorted = True)
            h = F.leaky_relu(h)
            #output
            h = self.dense(h)
        h_dict = to_hetero_feat(h, g.ndata['_TYPE'], G.ntypes)
        #Store edge weights
        attention_score[0] = g1.detach().numpy()
        attention_score[1] = g.edges()
        attention_score[2] = g.edata['_TYPE'].detach().numpy()
        attention_score[3] = g.nodes().detach().numpy()
        return h_dict

#Evaluate
def evaluate(model, graph, features, labels, index, BB_G, bb_features):
     model.eval()
     with torch.no_grad():
        logits = model(graph, BB_G, features, bb_features)
        logits = logits['node'][index]
        labels = labels[index]
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim =1)
        true = labels.argmax(dim =1)
        TP, TN, FP, FN = 0, 0, 0, 0
        e = 0.00000001
        for i in range(pred.shape[0]):
            if pred[i] == 0 and true[i] == 0:
                TP += 1
            if pred[i] == 1 and true[i] == 1:
                TN += 1
            if pred[i] == 0 and true[i] == 1:
                FP += 1
            if pred[i] == 1 and true[i] == 0:
                FN += 1
        P = TP/(TP + FP+ e)
        R = TP/(TP + FN+ e)
        F1 = (2 * P * R)/(P + R +e)
        acc = torch.sum(pred == true).item() * 1.0 / len(index)
        return  loss, acc, P, F1

import random
random.seed(2)

def train_model(train_data_node, train_data_edge, BB_info, features, labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BB_ins = features[:, 5] #Basic block number
    features = np.asarray(features, float) #The attribute of instruction

    #Standardization
    for i in range(features.shape[1]):
        features[ :, i] = noramlization(features[:, i])

    features = torch.FloatTensor(features).to(device)
    features = np.delete(features, [5,6], axis = 1) #Remove basic block  and function attributes

    #Build heterogeneous graph of instruction layer
    graph = dict()
    keys = list(train_data_edge.keys())
    for i in range(len(keys)):
        value = train_data_edge[keys[i]] #the type of edge
        str1 = list()
        str1.append('node')
        str1.append(edge_type[keys[i]])
        str1.append('node')
        head = [] 
        tail = [] 
        for j in range(len(value)):
            head.append(int(value[j][0]))
            tail.append(int(value[j][1]))
        graph[tuple(str1)]= (head, tail)
    G = dgl.heterograph(graph) 

    #Basic block information processing
    bb_nodes, bb_features, bb_edge = BB_info #node,attribute,edge
    bb_features = np.asarray(bb_features, float)    
    #Standardization
    for i in range(bb_features.shape[1]):
        bb_features[ :, i] = noramlization(bb_features[:, i])
    bb_features = torch.FloatTensor(bb_features).to(device) 
    #Build graph of basic block layer
    head = []
    tail = []
    for i in range(len(bb_edge)):
        head.append(int(bb_edge[i][0]))
        tail.append(int(bb_edge[i][1]))
    BB_G = dgl.graph((head, tail))

    #Processing labels
    labels = np.asarray(labels, float)
    T_index = [] #Positive sample of node serial number
    F_index = [] #Negative sample of node serial number
    final_labels = [0]*len(labels)
    for i in range(len(labels)):
        if labels[i] >= 0.25: #Error rate >= 0.25  is vulnerable
            final_labels[i] = [1,0]
            T_index.append(i) 
        elif labels[i] > -1: #invulnerable
            final_labels[i] = [0,1]
            F_index.append(i)
        else: #No fault injection instructions
            final_labels[i] = [0,0]
    
    #Generate training, verification and test sets
    index = []
    test_index = []
    for i in range(len(labels)):
        if labels[i] != -1:
            index.append(i)
        else:
            test_index.append(i)
    #Training set: 70% instructions
    T_train = random.sample(T_index, int(len(T_index)*0.8)) 
    F_train = random.sample(F_index, int(len(F_index)*0.8)) 
    idx_train = T_train + F_train
    random.shuffle(idx_train)
    idx_train = torch.LongTensor(idx_train)

    #Test Set: 
    idx_test = list((set(T_index) - set(T_train))) + list((set(F_index) - set(F_train)))
    random.shuffle(idx_test)
    
    #Validation Set, 20% of Test set
    idx_val = idx_test[0:int(len(idx_test)*0.2)]
    for i in range(len(idx_val)):
        idx_test.remove(idx_val[i]) #Remove validation set from test set

    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)

    #The label of error rate
    labels = torch.FloatTensor(np.asarray(final_labels, float)).to(device)

    feature = {}
    feature['node'] = features
    #Model initialization
    model = HeteroRGCN(3, 2, 5, 1, 5, 2, BB_ins).to(device)
    #optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_Acc = 0
    model.train()
    #train model of 3000 epochs
    for epoch in range(3000):
        opt.zero_grad()
        logits = model(G, BB_G, feature, bb_features)
        #Get the result of prediction
        logits = logits['node']
        #Calculate cross entropy loss only for marked nodes
        loss = F.cross_entropy(logits[idx_train], labels[idx_train]) 
        pred = logits.argmax(dim =1)
        true = labels.argmax(dim =1)
        #Calculation accuracy
        acc = torch.sum(pred[idx_train] == true[idx_train]).item() * 1.0 / len(idx_train)
        loss.backward()
        opt.step()
        #Validate every 5 epochs and choose the best model
        if epoch % 5 == 0:
            val_loss, val_acc, val_P, val_F1 = evaluate(model, G, feature, labels, idx_val, BB_G, bb_features)
            if epoch < 1000:
                continue
            elif val_acc >= best_Acc:
                best_Acc = val_acc
                torch.save(model, path1+ '\\model.pkl')
    #Load the best model
    best_model = torch.load(path1+ '\\model.pkl')
    path = path1 + '\\att.npy'
    np.save(path, attention_score) #Store weights for edges

    #Testing
    best_model.eval()
    with torch.no_grad():
        logits = best_model(G, BB_G, feature, bb_features)
        logits = logits['node'][idx_test]
        label_val = labels[idx_test]
        pred = logits.argmax(dim =1)
        true = label_val.argmax(dim =1)
        TP, TN, FP, FN = 0, 0, 0, 0
        e = 0.00000001
        for i in range(pred.shape[0]):
            if pred[i] == 0 and true[i] == 0:
                TP += 1
            if pred[i] == 1 and true[i] == 1:
                TN += 1
            if pred[i] == 0 and true[i] == 1:
                FP += 1
            if pred[i] == 1 and true[i] == 0:
                FN += 1
        P = TP/(TP + FP+ e)
        R = TP/(TP + FN+ e)
        F1 = (2 * P * R)/(P + R +e)
        acc = torch.sum(pred == true).item() * 1.0 / len(idx_test)
        print('test_ACC %.4f, test_Pre %.4f, test_F1 %.4f' % (
                    acc,
                    P,
                    F1,
            ))
        
    return
if __name__ == '__main__':
    t0 = time.time()
    
    args = parse_args() #Parameter initialization
    features , labels = load_feature_data() #Load the attributes and labels of instruction
    train_data_node , train_data_edge = load_train_data() #Load the nodes and edges of instruction
    BB_info = load_BB_info() #Load basic block information
    train_model(train_data_node, train_data_edge, BB_info, features, labels)#Conduct model training

