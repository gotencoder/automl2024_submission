import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from dgl.sampling import random_walk
import h5py
import numpy as np
from graph_tools import *
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from plotting_script import *
from my_metrics import *




#Set random seed
torch.manual_seed(0)
np.random.seed(0)
dgl.random.seed(0)


def Points2graph(points=None, n_neighbours = None, algorithm = None):

    NN = NearestNeighbors(n_neighbors=n_neighbours, algorithm=algorithm).fit(points)
    adj_matrix = NN.kneighbors_graph(points).toarray()

    # Replacing from_numpy_matrix by DiGraph
    graph = nx.DiGraph(np.array(adj_matrix))
    #graph = nx.from_numpy_matrix(adj_matrix)

    return graph, adj_matrix




#filename = 'fkdv_TW_long_t.h5' #'ks_mu_16_71_long_t.h5'#fkdv_TW_long_t.h5
filename='sg_plane_wave_norm.h5'
data = h5py.File(filename,'r')
#S_points = np.array(data['u'][-1000:])
#times = np.array(data['t'][-1000:,0])
S_points = np.array(data['u'][ data['t'][:,0] > 300 ])
times = np.array(data['t'][ data['t'][:,0] > 300,0 ])


n_neighbours = 20
algortihm = 'auto'
G, adj_matrix = Points2graph(points=S_points, n_neighbours=n_neighbours, algorithm=algortihm)
#S_points = min_max_scaler(S_points)
S_points_tensor = torch.tensor(S_points).to(torch.float32)


g = dgl.from_networkx(G)
g.ndata['feat'] = S_points_tensor


'''n_neighbours = 20
algortihm = 'auto'
G, adj_matrix = Points2graph(points=S_points, n_neighbours=n_neighbours, algorithm=algortihm)
S_points_tensor = torch.tensor(S_points).to(torch.float32)


g = dgl.from_networkx(G)
g.ndata['feat'] = S_points_tensor'''


# Define the GraphSAGE model with a link predictor module for negative sampling
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

# Create the GraphSAGE model and optimizer
#model = UGAT(3, 256, 2, neg_samples=1, num_heads=3, aggregation="concat")
model = UGAT(64, 256, 3, neg_samples=1, num_heads=1, aggregation="mean")
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
epoch = 1000

# Visualizing the architecture
print(model)

# Train the model with negative sampling loss
for epoch in range(epoch):
    h, pos_score, neg_score = model(g, g.ndata['feat'])
    loss = negative_sampling_loss(pos_score, neg_score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch {}, Loss {:.4f}'.format(epoch, loss.item()))

# Extract the learned node embeddings
with torch.no_grad():
    h, _, _ = model(g, g.ndata['feat'])
    h = h.numpy()
    h = h.reshape(h.shape[0],h.shape[-1])

metric = get_metrics(S_points,h,k=n_neighbours)
print(metric)
save_file = 'Savefile'
#single_plot_3d(h,times,'GAT',show_image=True,save_image=True,save_file=save_file)
