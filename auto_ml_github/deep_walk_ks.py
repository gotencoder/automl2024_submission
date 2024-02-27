from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.datasets import make_blobs
from sklearn import datasets
import networkx as nx
from sklearn.preprocessing import StandardScaler
import numpy as np
import h5py

from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from karateclub import DeepWalk
from graph_tools import Points2graph
#from plotting_script import *
from my_metrics import *



def min_max_scaler(data:np.ndarray): #Return transformed torch object
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Perform min-max scaling for each column
    data = (data - min_vals) / (max_vals - min_vals)
    
    return data



filename = 'ks_mu_16_71_long_t.h5' #'ks_mu_16_71_long_t.h5'#fkdv_TW_long_t.h5
#filename = 'sg_plane_wave_norm.h5'
data = h5py.File(filename,'r')
#S_points = np.array(data['u'][-1000:])
#times = np.array(data['t'][-1000:,0])
S_points = np.array(data['u'][ data['t'][:,0] > 300 ])
times = np.array(data['t'][ data['t'][:,0] > 300,0 ])

n_neighbours = 20
algortihm = 'auto'
G, adj_matrix = Points2graph(points=S_points, n_neighbours=n_neighbours, algorithm=algortihm)

# train model and generate embedding
model = DeepWalk(walk_length=50,epochs=10, dimensions=3, window_size=5)
model.fit(G)
embedding = model.get_embedding()
metric = get_metrics(S_points,embedding,k=n_neighbours)
print(metric)


#Plot results
save_file = 'Savefile'
#single_plot_3d(embedding,times,'DW',show_image=True,save_image=True,save_file=save_file)




