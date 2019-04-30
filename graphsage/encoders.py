import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists0, adj_lists1, aggregator0, aggregator1,
            num_sample=None,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists0 = adj_lists0
        self.adj_lists1 = adj_lists1
        self.aggregator0 = aggregator0
        self.aggregator1 = aggregator1
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator0.cuda = cuda
        self.aggregator1.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 3 * self.feat_dim))
        #init.xavier_uniform(self.weight)
        init.normal_(self.weight, mean=0, std=1)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        #print nodes
        #print "hahaha"
        #print [self.adj_lists0[int(node)] for node in nodes]
        neigh_feats0 = self.aggregator0.forward(nodes, [self.adj_lists0[int(node)] for node in nodes], self.feat_dim,
                self.num_sample)
        #print neigh_feats0
        #print nodes
        #print "hehehe"
        #print [self.adj_lists1[int(node)] for node in nodes]
        neigh_feats1 = self.aggregator1.forward(nodes, [self.adj_lists1[int(node)] for node in nodes], self.feat_dim, 
                self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            #print self_feats
            #print neigh_feats0
            #print neigh_feats1
            combined = torch.cat([self_feats, neigh_feats0, neigh_feats1], dim=1)
        else:
            combined = neigh_feats
        #print neigh_feats.shape
        #print self_feats
        #print combined
        combined = F.relu(self.weight.mm(combined.t()))
        #print self.weight
        return combined
