import numpy as np
import networkx as nx
import pickle
import os
import torch

from scipy.spatial.distance import cdist
from feature_extraction import extract_edge_features

##################################################
# SIGMA
##################################################

def sigma(dists, kth=8):

    knns = np.partition(
        dists,
        kth,
        axis=-1
    )[:, kth::-1]

    sig = knns.sum(axis=1).reshape(
        (knns.shape[0],1)
    ) / kth

    return sig + 1e-8

##################################################
# ADJ MATRIX
##################################################

def compute_adjacency_matrix_images(
        coord,
        feat,
        use_feat=False):

    coord = coord.reshape(-1,2)

    c_dist = cdist(coord,coord)

    if use_feat:

        f_dist = cdist(feat,feat)

        A = np.exp(
            -(c_dist/sigma(c_dist))**2
            -(f_dist/sigma(f_dist))**2
        )

    else:

        A = np.exp(
            -(c_dist/sigma(c_dist))**2
        )

    A = 0.5*A*A.T

    A[np.diag_indices_from(A)] = 0

    return A

##################################################
# EDGE LIST
##################################################

def compute_edges_list(
        A,
        kth=9):

    num_nodes = A.shape[0]

    new_kth = num_nodes-kth

    knns = np.argpartition(
        A,
        new_kth-1,
        axis=-1
    )[:,new_kth:-1]

    return knns

##################################################
# CIFAR SUPERPIXELS
##################################################

class CIFARSuperPix(torch.utils.data.Dataset):

    def __init__(
            self,
            data_dir,
            split='test',
            use_feat=False):

        with open(
            os.path.join(
                data_dir,
                f'cifar10_150sp_{split}.pkl'
            ),
            'rb'
        ) as f:

            self.labels,\
            self.sp_data = pickle.load(f)

        self.use_feat = use_feat

        self.img_size = 32

        self.Adj_matrices = []
        self.Graphs = []

        self.precompute()

    ################################################

    def precompute(self):

        for sample in self.sp_data:

            mean_px,\
            coord = sample[:2]

            coord = (
                coord/
                self.img_size
            )

            A = compute_adjacency_matrix_images(
                coord,
                mean_px,
                self.use_feat
            )

            self.Adj_matrices.append(A)

            ################################################
            # Graph
            ################################################

            G = nx.Graph()

            edges = compute_edges_list(A)

            for src,dsts in enumerate(edges):

                for dst in dsts:

                    if src != dst:

                        G.add_edge(
                            int(src),
                            int(dst)
                        )

            self.Graphs.append(G)

##################################################
# ATTACK
##################################################

def adversarial_attack(
        G,
        budget=0.10):

    G_attack = G.copy()

    edges = list(
        G_attack.edges()
    )

    n_attack = max(
        1,
        int(
            budget*
            len(edges)
        )
    )

    attacked_edges = []

    ################################################
    # DELETE
    ################################################

    remove_idx = np.random.choice(
        len(edges),
        n_attack,
        replace=False
    )

    for idx in remove_idx:

        edge = edges[idx]

        if G_attack.has_edge(*edge):

            G_attack.remove_edge(*edge)

            attacked_edges.append(edge)

    ################################################
    # INSERT
    ################################################

    nodes = list(
        G_attack.nodes()
    )

    count = 0

    while count < n_attack:

        u = np.random.choice(nodes)

        v = np.random.choice(nodes)

        if (
            u != v and
            not G_attack.has_edge(u,v)
        ):

            G_attack.add_edge(u,v)

            attacked_edges.append(
                (u,v)
            )

            count += 1

    return (
        G_attack,
        attacked_edges
    )

##################################################
# MAIN
##################################################

def main():

    X = []

    Y = []

    Graphs = []

    ################################################
    # DATASET
    ################################################

    dataset = CIFARSuperPix(
        data_dir=
        r"dataset\superpixels\superpixels",
        split='test',
        use_feat=False
    )

    ################################################
    # LOOP
    ################################################

    for idx in range(
            len(dataset.Graphs)
    ):

        G = dataset.Graphs[idx]

        label = dataset.labels[idx]

        Graphs.append(G)

        ################################################
        # ATTACK
        ################################################

        G_attack,\
        attack_edges = \
            adversarial_attack(
                G,
                budget=0.10
            )

        ################################################
        # FEATURES
        ################################################

        FE = extract_edge_features(
            G_attack
        )

        edge_list = list(
            G_attack.edges()
        )

        ################################################
        # STORE
        ################################################

        for i,e in enumerate(edge_list):

            X.append(FE[i])

            Y.append(label)

    ################################################
    # RETURN
    ################################################

    return (
        np.array(X),
        np.array(Y),
        Graphs
    )

