import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dgl.sampling import random_walk
import h5py
import numpy as np
from graph_tools import *
from plotting_script import *
from sklearn import datasets
from my_metrics import *
from matplotlib import gridspec



def min_max_scaler(data:np.ndarray): #Return transformed torch object
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Perform min-max scaling for each column
    data = (data - min_vals) / (max_vals - min_vals)
    #data = torch.tensor(data, dtype=torch.float32)

    return data


#Set random seed
torch.manual_seed(0)
np.random.seed(0)
dgl.random.seed(0)


#filename = 'ks_mu_16_71_long_t.h5' #'ks_mu_16_71_long_t.h5'#fkdv_TW_long_t.h5
filename = 'sg_plane_wave_norm.h5'
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



# Define the GraphSAGE model with a link predictor module for negative sampling
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
    

def negative_sampling_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    pos_loss = -torch.log(torch.clamp(torch.sigmoid(pos_score), min=1e-7, max=1-1e-7)) # Use clamp as nan values can arise due to numerical instability caused by large values in the exponential function when computing the sigmoid.
    neg_loss = -torch.log(torch.clamp(1 - torch.sigmoid(neg_score), min=1e-7, max=1-1e-7))
    loss = (pos_loss.sum() + neg_loss.sum()) / n_edges
    return loss


# Create the GraphSAGE model and optimizer
model = UGCN(64, 256, 3,neg_samples=1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
epoch = 1000

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

metric = get_metrics(S_points,h,k=n_neighbours)
print(metric)
save_file = 'Savefile'
#single_plot_3d(h,times,'GCN',show_image=False,save_image=True,save_file=save_file)
