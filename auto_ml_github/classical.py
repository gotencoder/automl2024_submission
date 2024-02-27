#Script to perform classical dimensionality reduction techniques on point cloud data
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, Isomap, MDS, SpectralEmbedding
import matplotlib.pylab as plt
from matplotlib import ticker
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh


#Define PCA algorithm
def PCA(data_input=np.array([]), n_dims=2):

    n, d = data_input.shape
    mean_val = np.tile(np.mean(data_input, 0), (n, 1)) #Create a centre of 'mass' point
    data_input = data_input - mean_val # Centre the data
    l, M = np.linalg.eig(np.dot(data_input.T, data_input)) #Find the eigenvalues 
    #(largest values correspond to principle componants) and eigenvectors 
    Y = np.dot(data_input, M[:, 0:n_dims])

    return Y #dimension of output will be (n,n_dims)



#Local linear embedding

def LLE(data_input=np.array([]), params={}):
	lle = LocallyLinearEmbedding(method="standard", **params)
	lle_data = lle.fit_transform(data_input)

	return lle_data



#Isomap: double check the use of the 'p' paramater

def ISOMAP(data_input=np.array([]), params={}):
	iso = Isomap(**params)
	iso_data = iso.fit_transform(data_input)

	return iso_data

#Multi-dimensional scaling

def Mu_Di_Sc(data_input=np.array([]), params={}):
	mds = MDS(**params)
	mds_data = mds.fit_transform(data_input)

	return mds_data

#Spectral Embedding 

def SE(data_input=np.array([]), params={}):
	se = SpectralEmbedding(**params)
	se_data = se.fit_transform(data_input)

	return se_data


        

        







    








 


