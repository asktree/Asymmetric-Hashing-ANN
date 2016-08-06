import numpy as np 
from scipy.cluster.vq import vq, kmeans2, whiten

def generate_hash_dict(to_hash, chunk_dim, minit='random'):
    #to_hash: m x n matrix w/ m observations

    M = to_hash.shape[0]
    N = to_hash.shape[1]

    if chunk_dim <= 0:
        raise ValueError("Chunk dimension must be strictly positive.")
    if chunk_dim > N or N % chunk_dim != 0:
        raise ValueError("Chunk dimension must divide N.")

    #Create empty arrays for out output. Labels are stored as bytes.
    #Slightly weird dimensionality due to spec
    centroids = np.zeros((N/chunk_dim, 256, chunk_dim))
    labels = np.zeros((M,N/chunk_dim),np.dtype('b'))

    #Perform k-means on each chunk to find centroids.
    for chunk in range(0, int(N/chunk_dim)):
        centroids[chunk,:,:], labels[:,chunk] = kmeans2(whiten(
            to_hash[:,chunk_dim*chunk:chunk_dim*(chunk+1)]),256,minit=minit)

    return (labels,centroids)