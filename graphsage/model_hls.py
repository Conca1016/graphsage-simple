import csv

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

import sys

import numpy as np
import time
import random
from random import shuffle
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, hls_dim):
        super(SupervisedGraphSage, self).__init__()
        self.xent = nn.MSELoss()
        
        #self.total_dim = emb_dim + hls_dim
        self.total_dim = hls_dim
        #self.total_dim = emb_dim
        self.weight1 = nn.Parameter(torch.FloatTensor(10, self.total_dim))
        init.normal_(self.weight1, mean=0, std=5)
        self.bn1 = nn.BatchNorm1d(10)        

        self.weight2 = nn.Parameter(torch.FloatTensor(1, 10))
        init.normal_(self.weight2, mean=0, std=5)
        

    def forward_fcn(self, hls_embs, funct_names):
        hls_emb_list = []
        for x in funct_names:
            hls_emb_list.append(torch.FloatTensor(np.asarray(hls_embs[x], dtype = float)))
        hls_embeds = torch.stack(hls_emb_list)
        #print hls_embeds
        #print hls_embeds, hls_embeds.shape, graph_embeds.shape
        #total_embeds = torch.cat([graph_embeds, hls_embeds], dim=1)
        total_embeds = hls_embeds
        #total_embeds = graph_embeds
        #print total_embeds
        #print self.weight
        #print self.weight1.mm(total_embeds.t())
        print "weight1:", self.weight1 
        emb_1 = F.sigmoid(self.bn1(self.weight1.mm(total_embeds.t()).t()).t())
        #print emb_1
        print "weight2", self.weight2
        emb_2 = self.weight2.mm(emb_1)
        #print emb_2
        return emb_2.t()


    def loss(self, funct_names, funct_map, hls_embs, ground_truth, data_size):
        #graph_embs = self.forward(funct_names, funct_map)
        #print graph_embs
        pred = self.forward_fcn(hls_embs, funct_names)
        #print pred
        #print scores.shape
        #print labels
        #print pred
        #print pred.view(data_size)
        #print ground_truth
        #print torch.FloatTensor(ground_truth).shape
        return self.xent(pred.view(data_size), ground_truth)

def load_cora(num_nodes, num_feats, dataset_dir):
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(dataset_dir + "node_feats.csv") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            #print info[-1]
            feat_data[i,:] = map(float, info[1:-1])
            #print feat_data[i, :]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
        #print feat_data

    adj_lists0 = defaultdict(set)
    adj_lists1 = defaultdict(set)
    with open(dataset_dir + "connect.csv") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists0[paper1].add(paper2)
            adj_lists1[paper2].add(paper1)
        #print adj_lists1
        for j in xrange(num_nodes):
            if j not in adj_lists0:
                adj_lists0[j].add(0)
            if j not in adj_lists1:
                adj_lists1[j].add(0)
             
    return feat_data, labels, adj_lists0, adj_lists1



def run_cora():
    np.random.seed(1)
    random.seed(1)
    ####num_mul = 100
    #print adj_lists1
    dataset_dir = "intel_dataset/"
   # features.cuda()
    
    '''
    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 14, 18, adj_lists, agg1, gcn=False, cuda=False)
    
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 15, adj_lists, agg2,
            base_model=enc1, gcn=False, cuda=False)
    agg3 = MeanAggregator(lambda nodes : enc2(nodes).t(), cuda=False)
    enc3 = Encoder(lambda nodes : enc2(nodes).t(), enc2.embed_dim, 12, adj_lists, agg3,
            base_model=enc2, gcn=False, cuda=False)
  
    agg4 = MeanAggregator(lambda nodes : enc3(nodes).t(), cuda=False)
    enc4 = Encoder(lambda nodes : enc3(nodes).t(), enc3.embed_dim, 11, adj_lists, agg4,
            base_model=enc3, gcn=False, cuda=False)
   
    enc1.num_samples = 5
    enc2.num_samples = 5
    enc3.num_samples = 5
    '''
   
    hls_dim = 1
    hls_test_col = -6
    graphsage = SupervisedGraphSage(hls_dim)
#    graphsage.cuda()

    funct_name = []
    funct_map = dict()
    with open(dataset_dir + "mulIndex.csv") as fp:
        for i,line in enumerate(fp):
            node_set = []
            info = line.strip().split()
            funct_name.append(info[-1]) 
            for j in xrange(len(info)-1):
                node_set.append(int(info[j])-1)
            funct_map[info[-1]] = node_set
    #print funct_name
    #print funct_map
    funct_hls_name = []
    hls_emb_map = dict()
    ground_truth = dict()
    with open(dataset_dir + "hls_graphsage.csv") as ff:
        for i, line in enumerate(ff):
            hls_emb = []
            y = line.strip().split()
            info = y[0].replace(',', ' ').split()
            if i == 0:
                continue
            else:
                funct_hls_name.append(info[-1])
                for j in xrange(len(info)-1):
                    if j < hls_dim:
                        hls_emb.append(float(info[j]))
                hls_emb_map[info[-1]] = hls_emb
                ground_truth[info[-1]] = float(info[hls_test_col])
    #print funct_hls_name
    #print hls_emb_map
    dataset_funct = []
    for x in funct_name:
        if x in funct_hls_name:
            dataset_funct.append(x)
    training_rate = 0.9
    random.shuffle(dataset_funct)
    train = dataset_funct[:int(len(dataset_funct) * training_rate)]
    test = dataset_funct[int(len(dataset_funct) * training_rate):]
    #print train
    #print test
   

    optimizer = torch.optim.SGD(graphsage.parameters(), lr=0.01)
    #optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.0001)
    #print filter(lambda p: p.requires_grad, graphsage.parameters())
    times = []
    epoch = 20
    batch_size = 20
    val_size = 20
    val_ind_list = []
    num_batch = len(train)//batch_size
    index_list = []
    loss_list = []
    val_list = []
    test_truth = []
    for x in test:
        test_truth.append(ground_truth[x])
    for e in xrange(epoch):
        shuffle(train)
        for i in xrange(num_batch):
            if i == 5:
                sys.exit()
            batch_graphs = train[i*batch_size: i*batch_size + batch_size]
            sub_ground_truth = []
            for x in batch_graphs:
                sub_ground_truth.append(ground_truth[x])
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_graphs, funct_map, hls_emb_map, Variable(torch.FloatTensor(np.asarray(sub_ground_truth))), batch_size)
            #print graphsage.weight2
            index_list.append(i+e*num_batch)
            loss_list.append(loss.item())
            print (i+e*num_batch), loss.item()
            loss.backward()
            optimizer.step()
            #enc3 = graph_forward(features, adj_lists)
            #print enc3.features
            #graphsage = SupervisedGraphSage(emb_dim, hls_dim, enc3)
            #optimizer.zero_grad()
            #print graphsage.enc.weight.grad
            #print graphsage.weight.grad
            print graphsage.weight1.grad
            print graphsage.weight2.grad
            end_time = time.time()
            times.append(end_time-start_time)
            if (i+e*num_batch) % val_size == 0:
                test_loss = graphsage.loss(test, funct_map, hls_emb_map, Variable(torch.FloatTensor(np.asarray(test_truth))), len(test))
                val_list.append(test_loss.item())
                val_ind_list.append(i+e*num_batch)
            #print batch, loss.item()
    #print graphsage.training
    #print val_list
    lines = plt.plot(index_list, loss_list, val_ind_list, val_list)
    plt.setp(lines[0], linewidth=2)
    plt.setp(lines[1], linewidth=2)
    plt.xlabel('batch')
    plt.ylabel('l2_loss')
    plt.legend(('Training loss', 'Validation loss'), loc='upper right')
    plt.title('Prediction based on HLS')
    plt.show()
    plt.savefig('training_loss.pdf')

    '''
    test_truth = []
    for x in test:
        test_truth.append(ground_truth[x])
    '''
    test_loss = graphsage.loss(test, funct_map, hls_emb_map, Variable(torch.FloatTensor(np.asarray(test_truth))), len(test))
    #print y_pred, y_test, val_output
    print "Test l2 loss:", test_loss.item()
    print "Average batch time:", np.mean(times)

     
    #test_output = graphsage.forward(test)
    #print "Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
if __name__ == "__main__":
    run_cora()
