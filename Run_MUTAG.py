import numpy as np
import networkx as nx

from torch_geometric.datasets import TUDataset

from feature_extraction import extract_edge_features

##################################################
# LOAD MUTAG
##################################################

dataset = TUDataset(
    root='./data',
    name='MUTAG'
)

##################################################
# EDGE INSERTION / DELETION ATTACK
##################################################

def adversarial_attack(
        G,
        budget=0.10):

    G_attack = G.copy()

    edges = list(G_attack.edges())

    n_attack = max(
        1,
        int(
            budget *
            len(edges)
        )
    )

    attacked_edges = []

    ################################################
    # Edge Deletion
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
    # Edge Insertion
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

            attacked_edges.append((u,v))

            count += 1

    return G_attack, attacked_edges

##################################################
# PYG GRAPH -> NETWORKX
##################################################

def pyg_to_networkx(data):

    G = nx.Graph()

    edge_index = data.edge_index.numpy()

    for i in range(
            edge_index.shape[1]
    ):

        u = int(edge_index[0,i])

        v = int(edge_index[1,i])

        G.add_edge(u,v)

    return G

##################################################
# MAIN
##################################################

def main():

    X = []

    Y = []

    Graphs = []

    ################################################
    # LOOP OVER MUTAG GRAPHS
    ################################################

    for graph_data in dataset:

        ################################################
        # ORIGINAL GRAPH
        ################################################

        G = pyg_to_networkx(
            graph_data
        )

        label = int(
            graph_data.y.item()
        )

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
        # FEATURE EXTRACTION
        ################################################

        FE = extract_edge_features(
            G_attack
        )

        edge_list = list(
            G_attack.edges()
        )

        ################################################
        # STORE FEATURES
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

