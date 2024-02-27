import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from dgl.sampling import random_walk
from scipy.sparse import spmatrix
import h5py
import numpy as np
from graph_tools import *
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from my_metrics import *
import random
from models import UGAT, UGCN, UGraphSAGE, negative_sampling_loss
from karateclub import DeepWalk
from scipy.sparse import csr_matrix
from classical import *
import time




#Set random seed
torch.manual_seed(0)
np.random.seed(0)
dgl.random.seed(0)
random.seed(0)


def get_key_for_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None




class AutoML_PC():
    def __init__(self, data: np.ndarray = None) -> None:
        self.data = data

    def Get_Hyperparam(self, seed=None) -> dict:
        random.seed(seed)
        hidden_feats_range = [128, 256, 512, 1024]  
        optimizer_choices = ['Adam', 'SGD'] 
        lr_range = np.linspace(1e-4,1e-2,5) 
        out_feats_range = [3, 4, 5] 
        walk_length_range = [50, 60, 70]  
        k_main_range = [40, 50, 60, 70]  
        walk_length_sub_range = [80, 90, 100]  
        
        # Selecting a random value within the specified range for each parameter
        hidden_feats = random.choice(hidden_feats_range)
        optimizer = random.choice(optimizer_choices)
        lr = random.choice(lr_range)  # Unpacking the tuple range
        out_feats = random.choice(out_feats_range)
        walk_length = random.choice(walk_length_range)
        k_main = random.choice(k_main_range)
        walk_length_sub = random.choice(walk_length_sub_range)
        
        temp_dict = {'lr': lr, 'hidden_feats': hidden_feats, 'optimizer': optimizer, 
                 'out_feats': out_feats, 'walk_length': walk_length,
                 'k_main': k_main, 'walk_length_sub': walk_length_sub
                }
        
        return temp_dict

   
    def Points_to_Graph(self, param_dict: dict = None) -> spmatrix:
        NN = NearestNeighbors(n_neighbors=param_dict['k_main'], algorithm='auto').fit(self.data)
        adj_matrix = NN.kneighbors_graph(self.data).toarray()
        graph = nx.DiGraph(np.array(adj_matrix))

        for i, node_features in enumerate(self.data):
            graph.nodes[i]['feat'] = torch.tensor(node_features).to(torch.float32)

        return graph, adj_matrix
    

    def Graph_to_Subgraph(self, graph: nx.Graph, param_dict: dict = None) -> nx.Graph:
        valid_subgraph = False
        attempts = 0
        max_attempts = 10  # Prevent infinite loop by setting a maximum number of attempts

        while not valid_subgraph and attempts < max_attempts:
            # Start from a random node
            current_node = random.choice(list(graph.nodes()))
            subgraph_nodes = {current_node}

            for _ in range(param_dict['walk_length_sub']):
                neighbors = [n for n in graph.neighbors(current_node) if n not in subgraph_nodes]
                if not neighbors:
                    break  # No more neighbors to traverse, try a new start

                # Move to a random neighbor
                current_node = random.choice(neighbors)
                subgraph_nodes.add(current_node)

            # Check if the subgraph meets the minimum size requirement
            print(f'Subraph is size {len(subgraph_nodes)}')
            if len(subgraph_nodes) >= param_dict['walk_length_sub']:
                valid_subgraph = True
            else:
                attempts += 1
                print(f"Attempt {attempts}: Subgraph too small, retrying...")

        if attempts == max_attempts:
            print("Maximum attempts reached, returning last subgraph despite size.")
        
        print('Subgraph nodes:', subgraph_nodes)
        subgraph = graph.subgraph(subgraph_nodes)
        subgraph_adj_matrix = nx.adjacency_matrix(subgraph)

        # Get the indexes for the subgraph nodes
        indexes = list(subgraph.nodes.keys())

        return subgraph, subgraph_adj_matrix, indexes

    
    def Evaluate_Qscore_PCA(self, param_dict: dict = None, sub_index: list = None, main = None) -> tuple:
        if main==False:
            result = np.real(PCA(self.data[sub_index], n_dims=param_dict['out_feats']))
            metric = get_metrics(self.data[sub_index],result,k=None)
        elif main==True:
            result = np.real(PCA(self.data, n_dims=param_dict['out_feats']))
            metric = get_metrics(self.data,result,k=None)
        else:
            raise Exception("Please give a True/False response to main graph variable")
        
        return metric
    
    
    def Evaluate_Qscore_LLE(self, param_dict: dict = None, sub_index: list = None, main = None) -> tuple:
        if main==False:
            params = {
            "n_neighbors": param_dict['walk_length_sub']-1,
            "n_components": param_dict['out_feats'],
            "n_jobs": -1,
            "random_state": 0}
            result = LLE(self.data[sub_index], params)
            metric = get_metrics(self.data[sub_index],result,k=None)
        elif main==True:
            params = {
            "n_neighbors": param_dict['k_main'],
            "n_components": param_dict['out_feats'],
            "n_jobs": -1,
            "random_state": 0}
            result = LLE(self.data, params)
            metric = get_metrics(self.data,result,k=None)
        else:
            raise Exception("Please give a True/False response to main graph variable")
        
        return metric
    
    def Evaluate_Qscore_ISOMAP(self, param_dict: dict = None, sub_index: list = None, main = None) -> tuple:
        if main==False:
            params = {
            "n_neighbors": param_dict['walk_length_sub']-1,
            "n_components": param_dict['out_feats'],
            "n_jobs": -1}
            result = ISOMAP(self.data[sub_index], params)
            metric = get_metrics(self.data[sub_index],result,k=None)
        elif main==True:
            params = {
            "n_neighbors": param_dict['k_main'],
            "n_components": param_dict['out_feats'],
            "n_jobs": -1}
            result = ISOMAP(self.data, params)
            metric = get_metrics(self.data,result,k=None)
        else:
            raise Exception("Please give a True/False response to main graph variable")
        
        return metric
    
    def Evaluate_Qscore_MDS(self, param_dict: dict = None, sub_index: list = None, main = None) -> tuple:
        if main==False:
            params = {
            "n_components": param_dict['out_feats'],
            "random_state": 0}
            result = Mu_Di_Sc(self.data[sub_index], params)
            metric = get_metrics(self.data[sub_index],result,k=None)
        elif main==True:
            params = {
            "n_components": param_dict['out_feats'],
            "random_state": 0}
            result = Mu_Di_Sc(self.data, params)
            metric = get_metrics(self.data,result,k=None)
        else:
            raise Exception("Please give a True/False response to main graph variable")
        
        return metric
    
    def Evaluate_Qscore_SE(self, param_dict: dict = None, sub_index: list = None, main = None) -> tuple:
        if main==False:
            params = {
            "n_neighbors": param_dict['walk_length_sub']-1,
            "n_components": param_dict['out_feats'],
            "n_jobs": -1,
            "random_state": 0}
            result = SE(self.data[sub_index], params)
            metric = get_metrics(self.data[sub_index],result,k=None)
        elif main==True:
            params = {
            "n_neighbors": param_dict['k_main'],
            "n_components": param_dict['out_feats'],
            "n_jobs": -1,
            "random_state": 0}
            result = SE(self.data, params)
            metric = get_metrics(self.data,result,k=None)
        else:
            raise Exception("Please give a True/False response to main graph variable")
        
        return metric


    def Evaluate_Qscore_UGAT(self, graph: spmatrix = None, param_dict: dict = None, main=None) -> np.array:
        g = dgl.from_networkx(graph, node_attrs=['feat'])
        if main==False:
            epoch = 100
            model = UGAT(64, param_dict['hidden_feats'], param_dict['out_feats'], neg_samples=1, num_heads=1, aggregation="mean", walk_length=param_dict['walk_length_sub'])
        elif main==True:
            epoch = 1000
            model = UGAT(64, param_dict['hidden_feats'], param_dict['out_feats'], neg_samples=1, num_heads=1, aggregation="mean", walk_length=param_dict['walk_length'])
        else:
            raise Exception("Please give a True/False response to main graph variable")

        if param_dict['optimizer']=='SGD':
            optimizer = torch.optim.SGD(model.parameters(),lr=param_dict['lr'])
        elif param_dict['optimizer']=='Adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=param_dict['lr'])
        else:
            raise Exception("Optimizer selected is not implemented")
        
        # Train the model with negative sampling loss
        for epoch in range(epoch):
            h, pos_score, neg_score = model(g, g.ndata['feat'])
            loss = negative_sampling_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Epoch {}, Loss {:.4f}'.format(epoch, loss.item()))
            
        # Extract the learned node embeddings
        with torch.no_grad():
            h, _, _ = model(g, g.ndata['feat'])
            h = h.numpy()
            h = h.reshape(h.shape[0],h.shape[-1])
            #print(h.shape)
        
        metric = get_metrics(self.data[np.array(graph.nodes)],h,k=None) #k does not need to be set here as we are not making the graph

        return metric
    

    def Evaluate_Qscore_UGCN(self, graph: spmatrix = None, param_dict: dict = None, main=None) -> tuple:
        g = dgl.from_networkx(graph, node_attrs=['feat'])
        if main==False:
            epoch = 100
            model = UGCN(64, param_dict['hidden_feats'], param_dict['out_feats'], neg_samples=1, walk_length=param_dict['walk_length_sub'])
        elif main==True:
            epoch = 1000
            model = UGCN(64, param_dict['hidden_feats'], param_dict['out_feats'], neg_samples=1, walk_length=param_dict['walk_length'])
        else:
            raise Exception("Please give a True/False response to main graph variable")

        model = UGCN(64, param_dict['hidden_feats'], param_dict['out_feats'], neg_samples=1)
        if param_dict['optimizer']=='SGD':
            optimizer = torch.optim.SGD(model.parameters(),lr=param_dict['lr'])
        elif param_dict['optimizer']=='Adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=param_dict['lr'])
        else:
            raise Exception("Optimizer selected is not implemented")
        
        # Train the model with negative sampling loss
        for epoch in range(epoch):
            h, pos_score, neg_score = model(g, g.ndata['feat'])
            loss = negative_sampling_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Epoch {}, Loss {:.4f}'.format(epoch, loss.item()))
            
        # Extract the learned node embeddings
        with torch.no_grad():
            h, _, _ = model(g, g.ndata['feat'])
            h = h.numpy()
        
        metric = get_metrics(self.data[np.array(graph.nodes)],h,k=None) #k does not need to be set here as we are not making the graph

        return metric
    

    def Evaluate_Qscore_UGraphSAGE(self, graph: spmatrix = None, param_dict: dict = None, main=None) -> tuple:
        g = dgl.from_networkx(graph, node_attrs=['feat'])
        if main==False:
            epoch = 100
            model = UGraphSAGE(64, param_dict['hidden_feats'], param_dict['out_feats'], neg_samples=1,aggregator_type="mean", walk_length=param_dict['walk_length_sub'])
        elif main==True:
            epoch = 1000
            model = UGraphSAGE(64, param_dict['hidden_feats'], param_dict['out_feats'], neg_samples=1, aggregator_type="mean", walk_length=param_dict['walk_length'])
        else:
            raise Exception("Please give a True/False response to main graph variable")

        if param_dict['optimizer']=='SGD':
            optimizer = torch.optim.SGD(model.parameters(),lr=param_dict['lr'])
        elif param_dict['optimizer']=='Adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=param_dict['lr'])
        else:
            raise Exception("Optimizer selected is not implemented")
        
        # Train the model with negative sampling loss
        for epoch in range(epoch):
            h, pos_score, neg_score = model(g, g.ndata['feat'])
            loss = negative_sampling_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Epoch {}, Loss {:.4f}'.format(epoch, loss.item()))
            
        # Extract the learned node embeddings
        with torch.no_grad():
            h, _, _ = model(g, g.ndata['feat'])
            h = h.numpy()
        
        metric = get_metrics(self.data[np.array(graph.nodes)],h,k=None) #k does not need to be set here as we are not making the graph

        return metric
    
    
    def Evaluate_Qscore_DW(self, graph: spmatrix = None, param_dict: dict = None, main=False) -> tuple:
        if main==False:
            epoch = 100
            mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, mapping)
            model = DeepWalk(walk_length=param_dict['walk_length_sub'],epochs=epoch, dimensions=param_dict['out_feats'], window_size=5)
        elif main==True:
            epoch = 1000
            model = DeepWalk(walk_length=param_dict['walk_length'],epochs=epoch, dimensions=param_dict['out_feats'], window_size=5)
        else:
            raise Exception("Please give a True/False response to main graph variable")
        
        model.fit(graph)
        embedding = model.get_embedding()
        metric = get_metrics(self.data[np.array(graph.nodes)],embedding,k=None) #Commented out assert 

        return metric
        

    def Optimize_Hyperparameters(self, n_iter: int, T: int) -> dict:
        #Loop through the list of dictionaries to evaluate
        q_temp = (0,0,0)
        optimal_params = {}
        final_q = 0
        optimal_method = None
        i=0
        for i in range(n_iter):
            params = self.Get_Hyperparam(seed=i)
            main_graph, _ = self.Points_to_Graph(params)
            sub_graph, _, sub_index = self.Graph_to_Subgraph(main_graph, params)
            q_scores_gat = self.Evaluate_Qscore_UGAT(sub_graph, params, main=False)
            q_scores_gcn = self.Evaluate_Qscore_UGCN(sub_graph, params, main=False)
            q_scores_gs = self.Evaluate_Qscore_UGraphSAGE(sub_graph, params, main=False)
            q_scores_dw = self.Evaluate_Qscore_DW(sub_graph,params, main=False)
            q_scores_pca = self.Evaluate_Qscore_PCA(params, sub_index ,main=False)
            q_scores_lle = self.Evaluate_Qscore_LLE(params, sub_index ,main=False)
            q_scores_mds = self.Evaluate_Qscore_MDS(params, sub_index ,main=False)
            q_scores_isomap = self.Evaluate_Qscore_ISOMAP(params, sub_index ,main=False)
            q_scores_se = self.Evaluate_Qscore_SE(params, sub_index ,main=False)
            print(q_scores_gat)
            print(q_scores_gcn)
            print(q_scores_gs)
            print(q_scores_dw)
            print(q_scores_pca)
            print(q_scores_lle)
            print(q_scores_mds)
            print(q_scores_isomap)
            print(q_scores_se)
            score_dict = {'PCA': q_scores_pca, 'MDS': q_scores_mds, 'LLE': q_scores_lle, 'ISOMAP': q_scores_isomap,
                          'SE': q_scores_se, 'GAT': q_scores_gat, 'GS': q_scores_gs, 'GCN': q_scores_gcn, 'DW': q_scores_dw}
            q_scores = max(q_scores_gat, q_scores_gcn, q_scores_gs, q_scores_dw, q_scores_pca, q_scores_lle, q_scores_mds, 
                           q_scores_isomap,q_scores_se ,key=lambda x: (T*x[0] + (1-T)*x[1]))
            best_method = get_key_for_value(score_dict, q_scores)
            #Optimize based on global q score value of subgraph
            if ((T*q_scores[0] + (1-T)*q_scores[1]) > (T*q_temp[0] + (1-T)*q_temp[1])):
                q_temp = q_scores
                optimal_params = params
                final_q = q_scores
                optimal_method = best_method
                        
        
        #Make final graph and eval q score -> We also need to find which method performs the best
        final_graph, _ = self.Points_to_Graph(optimal_params)
        final_q = None

        if optimal_method=='PCA':
            final_q = self.Evaluate_Qscore_PCA(optimal_params, None ,main=True)
            print(f'Optimal method is {optimal_method}')
        elif optimal_method=='MDS':
            final_q = self.Evaluate_Qscore_MDS(optimal_params, None ,main=True)
            print(f'Optimal method is {optimal_method}')
        elif optimal_method=='LLE':
            final_q = self.Evaluate_Qscore_LLE(optimal_params, None ,main=True)
            print(f'Optimal method is {optimal_method}')
        elif optimal_method=='ISOMAP':
            final_q = self.Evaluate_Qscore_ISOMAP(optimal_params, None ,main=True)
            print(f'Optimal method is {optimal_method}')
        elif optimal_method=='SE':
            final_q = self.Evaluate_Qscore_SE(optimal_params, None ,main=True)
            print(f'Optimal method is {optimal_method}')
        elif optimal_method=='GAT':
            final_q = self.Evaluate_Qscore_UGAT(final_graph, optimal_params, main=True)
            print(f'Optimal method is {optimal_method}')
        elif optimal_method=='GS':
            final_q = self.Evaluate_Qscore_UGraphSAGE(final_graph, optimal_params, main=True)
            print(f'Optimal method is {optimal_method}')
        elif optimal_method=='GCN':
            final_q = self.Evaluate_Qscore_UGCN(final_graph, optimal_params, main=True)
            print(f'Optimal method is {optimal_method}')
        elif optimal_method=='DW':
            final_q = self.Evaluate_Qscore_DW(final_graph, optimal_params, main=True)
            print(f'Optimal method is {optimal_method}')
        else:
            raise Exception("Please ensure graph method is valid")

        return final_q, optimal_params 
    
    def run_loop(self, model_name: str = None, n_iter: int = None):
        #This will run a given model for n_iter unique hyperparams
        #It will monitor the time and q_score statistics -> return these values
        i=0
        q_list = np.empty((n_iter,3), dtype=float)
        time_list = []
        begin_time = time.time()
        for i in range(n_iter):
            new_params = self.Get_Hyperparam(seed=random.randint(0, 10*n_iter))
            final_graph, _ = self.Points_to_Graph(new_params)
            if model_name=='PCA':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_PCA(new_params, None ,main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            elif model_name=='MDS':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_MDS(new_params, None ,main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            elif model_name=='LLE':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_LLE(new_params, None ,main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            elif model_name=='ISOMAP':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_ISOMAP(new_params, None ,main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            elif model_name=='SE':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_SE(new_params, None ,main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            elif model_name=='GAT':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_UGAT(final_graph, new_params, main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            elif model_name=='GS':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_UGraphSAGE(final_graph, new_params, main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            elif model_name=='GCN':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_UGCN(final_graph, new_params, main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            elif model_name=='DW':
                temp1_time = time.time()
                final_q = self.Evaluate_Qscore_DW(final_graph, new_params, main=True)
                temp2_time = time.time()
                time_list += [temp2_time-temp1_time]
                q_list[i] = final_q
                print(f'Optimal method is {model_name}')
            else:
                raise Exception("Please ensure graph method is valid")
            
        final_time = time.time()
        total_time = final_time - begin_time
        mean_time = np.mean(np.array(time_list))
        std_time = np.std(np.array(time_list))
        q_mean_values = q_list.mean(axis=0)
        q_std_values = q_list.std(axis=0)


        return q_mean_values, q_std_values, total_time, mean_time, std_time




    
filename = 'sg_plane_wave.h5' # "ks_mu_16_71_long_t.h5","fkdv_TW_long_t.h5", "sg_plane_wave.h5"
data = h5py.File(filename, 'r')
#S_points = np.array(data['u'][ data['t'][:,0] > 300 ]) 
S_points = np.array(data['u']) # Uncomment this line for the SG data
model = AutoML_PC(S_points)
hyperparam = model.Get_Hyperparam()
graph1, _ = model.Points_to_Graph(hyperparam)
start_time = time.time()
e1, e2= model.Optimize_Hyperparameters(10, T=0.5)
end_time = time.time()

#Run code below to obtain results for the individual model runs
q_mean, q_std, t_time, m_time, s_time = model.run_loop(model_name='DW', n_iter=10)
print(f"Total time is {end_time - start_time} seconds")











