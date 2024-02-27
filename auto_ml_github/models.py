import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from dgl.nn.pytorch.conv import GATConv, GraphConv
from dgl.sampling import random_walk
import numpy as np


class UGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, aggregator_type=None, neg_samples=1, dropout=0.0, walk_length=5):
        super(UGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type=aggregator_type) # Convolutional layer 1
        self.conv2 = SAGEConv(hidden_feats, hidden_feats*2, aggregator_type=aggregator_type) # Convolutional layer 2
        self.conv3 = SAGEConv(hidden_feats*2, out_feats, aggregator_type=aggregator_type) # Convolutional layer 3
        self.link_predictor = nn.Linear(out_feats * 2, 1)
        self.neg_samples = neg_samples
        self.dropout = dropout
        self.walk_length = walk_length

    def forward(self, g, x):
        h = F.relu(self.conv1(g, x))
        h = F.dropout(h, p=self.dropout)
        h = F.relu(self.conv2(g, h))
        h = F.dropout(h, p=self.dropout)
        h = self.conv3(g,h)
        h = F.dropout(h, p=self.dropout)
        g.ndata['h'] = h
        pos_score = self.compute_pos_score(g)
        neg_score = self.compute_neg_score(g)
        return h, pos_score, neg_score

    def compute_pos_score(self, g):
        # Compute similarity scores for positive pairs of nodes (connected in the graph)
        src, dst = g.edges()
        h_src, h_dst = g.ndata['h'][src], g.ndata['h'][dst]
        score = self.link_predictor(torch.cat([h_src, h_dst], dim=1))
        return score.squeeze()

    def compute_neg_score(self, g):
        # Compute similarity scores for negative pairs of nodes (disconnected in the graph)
        src, dst = self.generate_neg_samples(g)
        h_src, h_dst = g.ndata['h'][src], g.ndata['h'][dst]
        score = self.link_predictor(torch.cat([h_src, h_dst], dim=1))
        return score.squeeze()
    

    def generate_neg_samples(self, g):
        src, dst = g.edges()
        neg_src = []
        neg_dst = []
        for i in range(self.neg_samples): #Get a single negative source and destination pair
            rw = random_walk(g, nodes=torch.randint(high=g.num_nodes(), size=(src.shape[0],)), length=self.walk_length)
            neg_src.append(rw[0][:, 0])
            neg_dst.append(rw[0][:, -1])
        neg_src = torch.cat(neg_src)
        neg_dst = torch.cat(neg_dst)
        return neg_src, neg_dst


class UGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, neg_samples=1, dropout=0.0, walk_length=5):
        super(UGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats) # Convolutional layer 1
        self.conv2 = GraphConv(hidden_feats, hidden_feats*2) # Convolutional layer 2
        self.conv3 = GraphConv(hidden_feats*2, out_feats) # Convolutional layer 3
        self.link_predictor = nn.Linear(out_feats * 2, 1)
        self.neg_samples = neg_samples
        self.dropout = dropout
        self.walk_length = walk_length

    def forward(self, g, x):
        h = F.relu(self.conv1(g, x))
        h = F.dropout(h, p=self.dropout)
        h = F.relu(self.conv2(g, h))
        h = F.dropout(h, p=self.dropout)
        h = self.conv3(g,h)
        h = F.dropout(h, p=self.dropout)
        g.ndata['h'] = h
        pos_score = self.compute_pos_score(g)
        neg_score = self.compute_neg_score(g)
        return h, pos_score, neg_score

    def compute_pos_score(self, g):
        # Compute similarity scores for positive pairs of nodes (connected in the graph)
        src, dst = g.edges()
        h_src, h_dst = g.ndata['h'][src], g.ndata['h'][dst]
        score = self.link_predictor(torch.cat([h_src, h_dst], dim=1))
        return score.squeeze()

    def compute_neg_score(self, g):
        # Compute similarity scores for negative pairs of nodes (disconnected in the graph)
        src, dst = self.generate_neg_samples(g)
        h_src, h_dst = g.ndata['h'][src], g.ndata['h'][dst]
        score = self.link_predictor(torch.cat([h_src, h_dst], dim=1))
        return score.squeeze()
    

    def generate_neg_samples(self, g):
        src, dst = g.edges()
        neg_src = []
        neg_dst = []
        for i in range(self.neg_samples): #Get a single negative source and destination pair
            rw = random_walk(g, nodes=torch.randint(high=g.num_nodes(), size=(src.shape[0],)), length=self.walk_length)
            neg_src.append(rw[0][:, 0])
            neg_dst.append(rw[0][:, -1])
        neg_src = torch.cat(neg_src)
        neg_dst = torch.cat(neg_dst)
        return neg_src, neg_dst





class UGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats,
                 neg_samples=1, dropout=0.0, walk_length=5, num_heads=1, aggregation="mean"):

        super(UGAT, self).__init__()

        self.num_heads = num_heads

        self.conv1 = GATConv(in_feats, hidden_feats, num_heads = self.num_heads) # Convolutional layer 1
        self.conv2 = GATConv(hidden_feats, hidden_feats*2, num_heads = self.num_heads) # Convolutional layer 2
        self.conv3 = GATConv(hidden_feats*2, out_feats, num_heads = self.num_heads) # Convolutional layer 3

        if aggregation == "mean":
            self.link_predictor = nn.Linear(2*out_feats, 1)
        else:
            self.link_predictor = nn.Linear(num_heads*out_feats, 1)

        self.neg_samples = neg_samples
        self.dropout = dropout
        self.walk_length = walk_length

        if aggregation == "mean":
            self._aggregate = self._mean_aggregate
        else:
            self._aggregate = self._concat_aggregate

    def _mean_aggregate(self, x:torch.Tensor=None) -> torch.Tensor:

        return torch.mean(x, axis=1)

    def _concat_aggregate(self, x:torch.Tensor=None) -> torch.Tensor:

        return x.reshape((-1, ) + (np.prod(x.shape[1:]),))

    def aggregate(self, x):

        return self._aggregate(x)

    def forward(self, g, x):

        h = F.relu(self.conv1(g, x))
        h = F.dropout(h, p=self.dropout)
        h = self.aggregate(x=h)
        h = F.relu(self.conv2(g, h))
        h = F.dropout(h, p=self.dropout)
        h = self.aggregate(x=h)
        h = self.conv3(g,h)
        h = F.dropout(h, p=self.dropout)
        h = self.aggregate(x=h)

        g.ndata['h'] = h
        pos_score = self.compute_pos_score(g)
        neg_score = self.compute_neg_score(g)
        return h, pos_score, neg_score

    def compute_pos_score(self, g):
        # Compute similarity scores for positive pairs of nodes (connected in the graph)
        src, dst = g.edges()
        h_src, h_dst = g.ndata['h'][src], g.ndata['h'][dst]
        score = self.link_predictor(torch.cat([h_src, h_dst], dim=1))
        return score.squeeze()

    def compute_neg_score(self, g):
        # Compute similarity scores for negative pairs of nodes (disconnected in the graph)
        src, dst = self.generate_neg_samples(g)
        h_src, h_dst = g.ndata['h'][src], g.ndata['h'][dst]
        score = self.link_predictor(torch.cat([h_src, h_dst], dim=1))
        return score.squeeze()

    def generate_neg_samples(self, g):
        src, dst = g.edges()
        neg_src = []
        neg_dst = []
        for i in range(self.neg_samples): #Get a single negative source and destination pair
            rw = random_walk(g, nodes=torch.randint(high=g.num_nodes(), size=(src.shape[0],)), length=self.walk_length)
            neg_src.append(rw[0][:, 0])
            neg_dst.append(rw[0][:, -1])
        neg_src = torch.cat(neg_src)
        neg_dst = torch.cat(neg_dst)
        return neg_src, neg_dst

def negative_sampling_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    pos_loss = -torch.log(torch.clamp(torch.sigmoid(pos_score), min=1e-7, max=1-1e-7)) # Use clamp as nan values can arise due to numerical instability caused by large values in the exponential function when computing the sigmoid.
    neg_loss = -torch.log(torch.clamp(1 - torch.sigmoid(neg_score), min=1e-7, max=1-1e-7))
    loss = (pos_loss.sum() + neg_loss.sum()) / n_edges
    return loss