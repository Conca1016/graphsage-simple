import csv

import sys

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

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

    def __init__(self, emb_dim, hls_dim, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.MSELoss()

        self.weight = nn.Parameter(torch.FloatTensor(emb_dim, enc.embed_dim))
        init.normal_(self.weight, mean=0, std=1)
        
        self.total_dim = emb_dim + hls_dim
        #self.total_dim = hls_dim
        #self.total_dim = emb_dim
        self.weight1 = nn.Parameter(torch.FloatTensor(10, self.total_dim))
        init.normal_(self.weight1, mean=0, std=1)
        self.bn1 = nn.BatchNorm1d(10)      
  
        self.weight2 = nn.Parameter(torch.FloatTensor(1, 10))
        init.normal_(self.weight2, mean=0, std=1)


    def forward(self, funct_names, funct_map, embed=False):
        emb_list = []
        #print self.enc
        #print self.enc.base_model.aggregator(funct_map['7_136'])
        for x in funct_names:
            #print len(funct_map[x])
            emb = self.enc(funct_map[x])
            #print len(funct_map[x])
            emb = torch.sum(emb, dim=1)
            #print emb
            emb_list.append(emb)
        embeds = torch.stack(emb_list)
        #print embeds
        #graph_embeds = F.sigmoid(self.weight.mm(embeds.t()))
        graph_embeds = F.relu(self.weight.mm(embeds.t()))
        #graph_embeds = self.weight.mm(embeds.t())
        #print graph_embeds.shape
        #graph_embeds = graph_embeds/(torch.max(torch.abs(graph_embeds)).item())
        graph_embeds = graph_embeds/50
        if embed:
            embed_list = []
            embed_name = []
            for i in xrange(15):
                embed_name.append("emb%d" % i)
            embed_name.append("Design_Name")
            embed_list.append(embed_name)
            cnt = 0
            for x in funct_names:
                embed_entry = graph_embeds.t().tolist()[cnt] + [x]
                embed_list.append(embed_entry)
                cnt += 1
            with open("emb.csv", "w") as ff:
                writer = csv.writer(ff, delimiter = ',')
                writer.writerows(embed_list)
                
        #print funct_names
        #print graph_embeds.t()
        
        return graph_embeds.t()

    def forward_fcn(self, hls_embs, funct_names, graph_embeds):
        hls_emb_list = []
        for x in funct_names:
            hls_tensor = torch.FloatTensor(np.asarray(hls_embs[x], dtype = float))
            hls_tensor = hls_tensor/69
            #hls_tensor = hls_tensor/(torch.max(torch.abs(hls_tensor)).item())
            hls_emb_list.append(hls_tensor)
        hls_embeds = torch.stack(hls_emb_list)
        #print hls_embeds
        #print hls_embeds, hls_embeds.shape, graph_embeds.shape
        total_embeds = torch.cat([graph_embeds, hls_embeds], dim=1)
        #total_embeds = hls_embeds
        #total_embeds = graph_embeds
        #print total_embeds
        #print self.weight
        #print self.weight1
        #emb_1 = F.sigmoid(self.bn1(self.weight1.mm(total_embeds.t()).t()).t())
        emb_1 = F.relu(self.weight1.mm(total_embeds.t()))
        #print emb_1
        #print self.weight2
        emb_2 = self.weight2.mm(emb_1)
        #print emb_2
        return emb_2.t()


    def loss(self, funct_names, funct_map, hls_embs, ground_truth, data_size):
        graph_embs = self.forward(funct_names, funct_map)
        #print graph_embs
        pred = self.forward_fcn(hls_embs, funct_names, graph_embs)
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


def graph_forward(node_feat_dim, features, adj_lists0, adj_lists1, agg_sel=False):
    agg1_0 = MeanAggregator(features, cuda=False)
    agg1_1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, node_feat_dim, 15, adj_lists0, adj_lists1, agg1_0, agg1_1, gcn=agg_sel, cuda=False)

    agg2_0 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    agg2_1 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 12, adj_lists0, adj_lists1, agg2_0, agg2_1,
            base_model=enc1, gcn=agg_sel, cuda=False)

    agg3_0 = MeanAggregator(lambda nodes : enc2(nodes).t(), cuda=False)
    agg3_1 = MeanAggregator(lambda nodes : enc2(nodes).t(), cuda=False)
    enc3 = Encoder(lambda nodes : enc2(nodes).t(), enc2.embed_dim, 12, adj_lists0, adj_lists1, agg3_0, agg3_1,
            base_model=enc2, gcn=agg_sel, cuda=False)
    return enc2

def run_cora():
    np.random.seed(1)
    random.seed(1)
    ####num_mul = 100
    #print adj_lists1
    num_nodes = 3597
    node_feat_dim = 14
    dataset_dir = "vivado_dataset/"
    feat_data, labels, adj_lists0, adj_lists1 = load_cora(num_nodes, node_feat_dim, dataset_dir)
    features = nn.Embedding(num_nodes, node_feat_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()
    
    enc3 = graph_forward(node_feat_dim, features, adj_lists0, adj_lists1)
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
   
    emb_dim = 15
    hls_dim = 1
    hls_test_col = -3
    graphsage = SupervisedGraphSage(emb_dim, hls_dim, enc3)
#    graphsage.cuda()

    funct_name_train = []
    funct_name_test = []
    funct_map = dict()
    with open(dataset_dir + "mulIndexTest.csv") as fp:
        for i,line in enumerate(fp):
            node_set = []
            info = line.strip().split()
            funct_name_train.append(info[-1]) 
            for j in xrange(len(info)-1):
                node_set.append(int(info[j]))
            funct_map[info[-1]] = node_set
    with open(dataset_dir + "mulIndexTrain.csv") as fp:
        for i,line in enumerate(fp):
            node_set = []
            info = line.strip().split()
            funct_name_test.append(info[-1]) 
            for j in xrange(len(info)-1):
                node_set.append(int(info[j]))
            funct_map[info[-1]] = node_set
    #print funct_name
    #print funct_map
    funct_hls_name = []
    hls_emb_map = dict()
    ground_truth = dict()
    with open(dataset_dir + "hls_train.csv") as ff:
        for i, line in enumerate(ff):
            hls_emb = []
            y = line.strip().split()
            info = y[0].replace(',', ' ').split()
            if i == 0:
                continue
            else:
                funct_hls_name.append(info[-1])
                for j in xrange(len(info)-1):
                    if j < 5+hls_dim and j > 4:
                        hls_emb.append(float(info[j]))
                hls_emb_map[info[-1]] = hls_emb
                ground_truth[info[-1]] = float(info[hls_test_col])
    with open(dataset_dir + "hls_test.csv") as ff:
        for i, line in enumerate(ff):
            hls_emb = []
            y = line.strip().split()
            info = y[0].replace(',', ' ').split()
            if i == 0:
                continue
            else:
                funct_hls_name.append(info[-1])
                for j in xrange(len(info)-1):
                    if j < 5+hls_dim and j > 4:
                        hls_emb.append(float(info[j]))
                hls_emb_map[info[-1]] = hls_emb
                ground_truth[info[-1]] = float(info[hls_test_col])
    #print funct_hls_name
    #print hls_emb_map
    dataset_funct_train = []
    dataset_funct_test = []
    for x in funct_name_train:
        if x in funct_hls_name:
            dataset_funct_train.append(x)
    for x in funct_name_test:
        if x in funct_hls_name:
            dataset_funct_test.append(x)
    ###training_rate = 0.9
    ###random.shuffle(dataset_funct)
    train = dataset_funct_train[:]
    val_all = dataset_funct_test[:]
    test = dataset_funct_test[:]
    #print train
    #print test
   

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600], gamma=0.1)
    times = []
    epoch = 1000
    batch_size = 20
    val_size = 200
    val_ind_list = []
    num_batch = len(train)//batch_size
    index_list = []
    loss_list = []
    val_list = []
    test_truth = []
    for x in test:
        test_truth.append(ground_truth[x])
    for e in xrange(epoch):
        scheduler.step()
        shuffle(train)
        for i in xrange(num_batch):
            #if i == 10:
                #sys.exit()
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
            #print graphsage.weight1.grad
            #print graphsage.weight2.grad
            end_time = time.time()
            times.append(end_time-start_time)
            val_truth = []
            if (i+e*num_batch) % val_size == 0:
                val = val_all[:]
                for yy in val:
                    val_truth.append(ground_truth[yy])
                #random.shuffle(val_all)
                val_loss = graphsage.loss(val, funct_map, hls_emb_map, Variable(torch.FloatTensor(np.asarray(val_truth))), len(val))
                val_list.append(val_loss.item())
                val_ind_list.append(i+e*num_batch)
                print 'validation loss:    ', val_loss.item()
    #print graphsage.training
    #print val_list
    lines = plt.plot(index_list, loss_list, val_ind_list, val_list)
    plt.setp(lines[0], linewidth=2)
    plt.setp(lines[1], linewidth=2)
    plt.xlabel('batch')
    plt.ylabel('l2_loss')
    plt.legend(('Training loss', 'Validation loss'), loc='upper right')
    plt.title('#LUTs Prediction based on HLS+Graph_Learning')
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

    graph_embed = graphsage.forward(train+test, funct_map, embed=True)
     
    #test_output = graphsage.forward(test)
    #print "Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
if __name__ == "__main__":
    run_cora()
