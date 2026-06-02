import numpy as np
import networkx as nx
from scipy.stats import linregress


# =====================================================
# E1 TOTAL CONNECTIONS
# Eq (2)
# =====================================================

def total_connections(G):

    C = G.number_of_nodes()

    E1 = C*(C-1)/2

    return E1

# =====================================================
# E2 EDGE WEIGHT
# =====================================================

def edge_weight(G):

    weights = []

    for u,v in G.edges():

        deg_u = G.degree(u)
        deg_v = G.degree(v)

        w = deg_u + deg_v

        weights.append(w)

    return np.array(weights)

# =====================================================
# E3 TOTAL COMMON NEIGHBOR
# Eq (3)
# =====================================================

def common_neighbors_feature(G):

    vals = []

    for u,v in G.edges():

        cn = len(list(nx.common_neighbors(G,u,v)))

        vals.append(cn)

    return np.array(vals)

# =====================================================
# E4 EDGE RANK
# Eq (4)
# cosine similarity
# =====================================================

def edge_rank(G):

    vals = []

    for u,v in G.edges():

        fu = np.array([G.degree(u)])
        fv = np.array([G.degree(v)])

        sim = np.dot(fu,fv)/(np.linalg.norm(fu)*np.linalg.norm(fv)+1e-10)

        vals.append(sim)

    return np.array(vals)

# =====================================================
# E5 EDGE BETWEENNESS CENTRALITY
# Eq (5)
# =====================================================

def edge_betweenness(G):

    bc = nx.edge_betweenness_centrality(G)

    vals = []

    for edge in G.edges():

        vals.append(bc[edge])

    return np.array(vals)

# =====================================================
# E6 POWER LAW EXPONENT
# Eq (6)
# =====================================================

def power_law_feature(G):

    deg = np.array([d for n,d in G.degree()])

    deg = deg[deg > 0]

    x = np.log(np.arange(1,len(deg)+1))
    y = np.log(np.sort(deg)[::-1])

    slope,_,_,_,_ = linregress(x,y)

    alpha = abs(slope)

    return np.full(G.number_of_edges(), alpha)

# =====================================================
# E7 JACCARD SIMILARITY
# Eq (7)
# =====================================================

def jaccard_feature(G):

    vals = []

    for u,v,p in nx.jaccard_coefficient(G,G.edges()):

        vals.append(p)

    return np.array(vals)

# =====================================================
# E8 EDGE CONVOLUTION
# Eq (8)
# =====================================================

def edge_convolution(G):

    vals = []

    for u,v in G.edges():

        neigh_u = set(G.neighbors(u))
        neigh_v = set(G.neighbors(v))

        conv = len(neigh_u.union(neigh_v))

        vals.append(conv)

    return np.array(vals)

# =====================================================
# FEATURE EXTRACTION BLOCK (FIGURE 2)
# =====================================================

def extract_edge_features(G):

    E1 = total_connections(G)

    E2 = edge_weight(G)

    E3 = common_neighbors_feature(G)

    E4 = edge_rank(G)

    E5 = edge_betweenness(G)

    E6 = power_law_feature(G)

    E7 = jaccard_feature(G)

    E8 = edge_convolution(G)

    E1 = np.full(len(E2), E1)

    FE = np.column_stack([
        E1,
        E2,
        E3,
        E4,
        E5,
        E6,
        E7,
        E8
    ])

    return FE

