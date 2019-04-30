import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, feature_dim, num_sample=None):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
            #print samp_neighs

        if self.gcn:
        # consider node itself
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        #print unique_nodes_list
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = np.zeros((len(samp_neighs), len(unique_nodes)))
        ###mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        
        flag = 0
        if 0 in unique_nodes_list:
            flag = 1
        # select indecies of the nodes who are connected in the sample set
        column_indices = []
        for samp_neigh in samp_neighs:
            for n in samp_neigh:
                column_indices.append(unique_nodes[n])
                if n == 0:
                    remove_col = unique_nodes[n]
                    #print remove_col
        ###column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        #print mask
        # how many nodes deleted
        num = mask.size
        if flag == 1:
            mask = np.delete(mask, remove_col, 1)            
        #print mask
        #print mask.size
        if mask.size == 0:
            return torch.zeros([num, feature_dim], dtype=torch.float32)
        else:
            mask = Variable(torch.FloatTensor(mask))
            if self.cuda:
                mask = mask.cuda()
  
            # normalization
            num_neigh = mask.sum(1, keepdim=True)
            # avoid nan
            num_neigh[num_neigh==0] = 1
            mask = mask.div(num_neigh)
        
            #print unique_nodes_list 
            if 0 in unique_nodes_list:
                unique_nodes_list.remove(0)
            #print unique_nodes_list
            # select feature matrix of nodes in sample set
            if len(unique_nodes_list) == 0:
                return torch.zeros([1, feature_dim], dtype=torch.float32)
            else: 
                if self.cuda:
                    embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
                else:
                #print "xixixi"
                    embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
                #print "yichiyichi"
                #print mask
        
            # message passing
                to_feats = mask.mm(embed_matrix)
                return to_feats
