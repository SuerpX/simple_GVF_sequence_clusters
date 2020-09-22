import torch
from tqdm import tqdm
import numpy as np
import time

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

if use_cuda:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    
def K_means(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud, 
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
#     c = x[:K, :].clone()  # Simplistic random initialization
    c = x[torch.randperm(x.size()[0])[:K], :].clone()
    x_i = x[:, None, :]  # (Npoints, 1, D)
    start = time.time()
    for i in range(Niter):
    
        c_j = c[None, :, :]  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        D_ij_min, cl = D_ij.min(dim=1)  # Points -> Nearest cluster
        cl = cl.view(-1)
        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

        varience = 0
        for i in range(K):
            varience += torch.sum(D_ij_min[cl == i] ** 2)
        varience /= K
    end = time.time()
    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format( 
                Niter, end - start, Niter, (end-start) / Niter))
        print("Total Varience: ", varience)

    return cl, c, varience

def cluster(GVF_seq_dataset, k = 10, it = 10, verbose = False):
    min_varience = float("inf")
    for i in tqdm(range(it)):
        classes, temp_k_means_vec, varience = K_means(GVF_seq_dataset, k, verbose = verbose)
        if min_varience > varience:
            min_varience = varience
            k_means_vec = temp_k_means_vec
    return k_means_vec, min_varience

def KNN_clustering(GVF_seq, k_means):
    classes = torch.zeros(len(GVF_seq)).cuda()
    for i, n in enumerate(GVF_seq):
        dists = torch.cdist(n.unsqueeze(0), k_means)
        classes[i] = dists.min(1)[1]
    return classes


# def K_means(train_data, k):
#     GVF_dim = len(train_data[0])
#     k_means = torch.rand((k, GVF_dim)).cuda()
# #     print(k_means)
#     same_num = 0
# #     pbar = tqdm(total = k * GVF_dim, desc = "Clustering")
#     while True:
#         classes = torch.zeros(len(train_data)).cuda()
#         for i, n in enumerate(train_data):
#             dists = torch.cdist(n.unsqueeze(0), k_means)
# #             print(dists)
#             value, idx = dists.min(1)
# #             print(dists, idx, value)
#             classes[i] = idx
#         new_k_means = torch.zeros((k, GVF_dim)).cuda()
# #         print(classes)

#         for i in range(k):
#             if train_data[classes == i].size()[0] == 0:
#                 new_k_means[i] = k_means[i]
#             else:
#                 new_k_means[i] = torch.mean(train_data[classes == i], dim = 0)
# #         print(torch.sum(k_means == new_k_means))
        
#         new_same_num = int(torch.sum(k_means == new_k_means))
# #         print(new_same_num)
# #         pbar.update(new_same_num - same_num)
#         same_num = new_same_num
#         if torch.sum(k_means == new_k_means) == k * GVF_dim:
#             count = 0
#             varience = 0
#             for i in range(k):
#                 if train_data[classes == i].size()[0] != 0:
#                     varience += torch.var(train_data[classes == i])
#                     count += 1
#             varience /= count
#             break
        
#         k_means = new_k_means
        
# #         input()
#     return new_k_means, varience