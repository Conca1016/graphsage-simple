import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        #print scores.shape
        #print labels
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 5023
    num_feats = 14
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("hls/node_feats.csv") as fp:
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

    adj_lists = defaultdict(set)
    with open("hls/connect.csv") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_mul = 575
    feat_data, labels, adj_lists = load_cora()
    #print adj_lists
    features = nn.Embedding(5023, 14)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 14, 10, adj_lists, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 12, adj_lists, agg2,
            base_model=enc1, gcn=False, cuda=False)
    agg3 = MeanAggregator(lambda nodes : enc2(nodes).t(), cuda=False)
    enc3 = Encoder(lambda nodes : enc2(nodes).t(), enc2.embed_dim, 12, adj_lists, agg3,
            base_model=enc2, gcn=False, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5
    enc3.num_samples = 5

    graphsage = SupervisedGraphSage(3, enc3)
#    graphsage.cuda()

    with open("hls/mulIndex.csv") as fp:
        for i,line in enumerate(fp):
            if i == 0:
                mul_over = line.strip().split()
                mul_over = map(int, mul_over)
            elif i == 1:
                mul_under = line.strip().split()
                mul_under = map(int, mul_under)
            elif i == 2:
                mul_normal = line.strip().split()
                mul_normal = map(int, mul_normal)
    
    ######rand_indices = np.random.permutation(num_mul)
    np.random.shuffle(mul_over)
    np.random.shuffle(mul_under)
    np.random.shuffle(mul_normal)
    #test = mul_over[30:] + mul_under[60:] + mul_normal[195:]
    val = mul_over[50:] + mul_under[100:] + mul_normal[200:]
    train = list(mul_over[:50] + mul_under[:100] + mul_normal[:200])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    ####lambda1 = lambda batch: 0.7 if batch < 1000 else 0.7
    ####scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    for batch in range(1500):
        ####scheduler.step()
        ####if batch > 1000:
          ####train = list(mul_over[:50])
        batch_nodes = train[:30]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        #print Variable(torch.LongTensor(labels[np.array(batch_nodes)]))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.item()

    val_output = graphsage.forward(val)
    y_pred = val_output.data.numpy().argmax(axis=1)
    y_test = np.asarray(labels[val]).ravel()
    #print y_pred, y_test, val_output
    class_names = np.asarray(['Normal', 'OverEstimate', 'UnderEstimate'])
    plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')
    plt.show()
    plt.savefig('confusion')
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)

    #test_output = graphsage.forward(test)
    #print "Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
if __name__ == "__main__":
    run_cora()
