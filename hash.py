import numpy as np 
from numpy import transpose
from scipy.cluster.vq import vq, kmeans2, whiten
from scipy.optimize import leastsq
import math

def generate_hash_dict(to_hash, chunk_dim, save_to):
    #to_hash: R^(N x M); save_to: string; chunk_dim: int =/= 0
    to_hash = transpose(to_hash)
    N = to_hash.shape[0]
    M = to_hash.shape[1]

    #perform k-means on each chunk_dim chunk to find centroids
    num_chunks = math.ceil(N/chunk_dim)
    centroids = np.empty((num_chunks,256,N))
    closest_centroids = np.empty((num_chunks,M),int)
    for chunk in range(0,num_chunks):
        whitened = whiten(transpose(to_hash[chunk_dim*chunk:chunk_dim*(chunk+1),:]))
        out = kmeans2(whitened,256)
        centroids[chunk,:,:] = out[0]
        closest_centroids[chunk,:] = out[1]

    #switch to spock's output spec
    return (transpose(closest_centroids),centroids)