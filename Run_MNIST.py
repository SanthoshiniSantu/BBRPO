import numpy as np
import networkx as nx

from torchvision.datasets import MNIST
from torchvision import transforms


from feature_extraction import extract_edge_features

##################################################
# LOAD MNIST
##################################################

dataset = MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

##################################################
# IMAGE -> GRAPH
##################################################

def image_to_graph(img):

    img = img.squeeze().numpy()

    h,w = img.shape

    G = nx.Graph()

    for r in range(h):

        for c in range(w):

            idx = r*w+c

            G.add_node(
                idx,
                intensity=float(img[r,c])
            )

            if c < w-1:
                G.add_edge(idx,idx+1)

            if r < h-1:
                G.add_edge(idx,idx+w)

    return G

##################################################
# EDGE INSERTION / DELETION
##################################################

def adversarial_attack(
        G,
        budget=0.10):

    G_attack = G.copy()

    edges = list(G_attack.edges())

    n_attack = int(
        budget*len(edges)
    )

    ################################################
    # deletion
    ################################################

    remove_idx = np.random.choice(
        len(edges),
        n_attack,
        replace=False
    )

    attacked_edges = []

    for idx in remove_idx:

        u,v = edges[idx]

        if G_attack.has_edge(u,v):

            G_attack.remove_edge(u,v)

            attacked_edges.append((u,v))

    ################################################
    # insertion
    ################################################

    nodes = list(G_attack.nodes())

    count = 0

    while count < n_attack:

        u = np.random.choice(nodes)
        v = np.random.choice(nodes)

        if u!=v and not G_attack.has_edge(u,v):

            G_attack.add_edge(u,v)

            attacked_edges.append((u,v))

            count += 1

    return G_attack,attacked_edges

##################################################
# PREPARE DATASET
##################################################
def main():

    X = []
    Y = []
    Graphs = []

    for idx in range(1000):

        image,label = dataset[idx]

        ################################################
        # graph
        ################################################

        G = image_to_graph(image)
        Graphs.append(G)

        ################################################
        # attack
        ################################################

        G_attack,attack_edges = adversarial_attack(
            G,
            budget=0.10
        )

        ################################################
        # feature extraction
        ################################################

        FE = extract_edge_features(
            G_attack
        )

        edge_list = list(
            G_attack.edges()
        )

        ################################################
        # labels
        ################################################

        for i,e in enumerate(edge_list):

            X.append(FE[i])

            if e in attack_edges:

                Y.append(1)

            else:

                Y.append(0)

    return np.array(X), np.array(Y), Graphs

