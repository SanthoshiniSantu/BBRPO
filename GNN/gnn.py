import networkx as nx
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Layer
import numpy as np
import tensorflow as tf

from keras.optimizers import Adam as opt
tf.config.run_functions_eagerly(True)
import cv2


def attack(G):

    att = []
    for i, edge in enumerate(G.edges()):
        no_attack = [(u, v) for u,v in G.edges]
        if edge not in no_attack:
            data = {}


            H = G.copy()

            # attack an edge
            H.remove_edges_from(ebunch=[edge])

            n = len(G.nodes)
            retain_node_ids = [9, 3]
            H.add_edges_from([(u, v) for u in retain_node_ids for v in (n+1, n+2)])

            # remove nodes with degree < 2
            H = nx.k_core(H, k=2)
            H.remove_nodes_from([n + 1, n + 2])


            # delete connected nodes and edges
            diff_nodes = set(G.nodes()).difference(H.nodes())
            diff_edges = {e for e in G.edges() for n in diff_nodes if n in e}



            data['diff_nodes'] = list(diff_nodes)
            data['diff_edges'] = list(diff_edges)
            data['edge'] = edge
            att.append(True)
        else:
            att.append(False)
        ASR = len(G.edges) / att.count(False)
        if True in att:

            return True ,ASR
        else:
            return False, ASR


class GraphConvolution(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(GraphConvolution, self).build(input_shape)

    def call(self, x):
        output = tf.matmul(x, self.kernel)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split

def classify(
        data,
        target,
        tr_per,
        G,
        ACC,
        ASR,
        PRE):

    ################################################
    # DATA
    ################################################

    X = np.asarray(data)

    if X.ndim == 1:
        X = X.reshape(1,-1)

    y = np.asarray(target)

    if y.ndim == 0:
        y = np.array([y])

    ################################################
    # TRAIN TEST SPLIT
    ################################################

    if len(X) > 1:

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            train_size=tr_per,
            random_state=42
        )

    else:

        X_train = X
        X_test = X

        y_train = y
        y_test = y

    ################################################
    # PARAMETERS
    ################################################

    n_features = X_train.shape[1]

    n_hidden = 128

    n_classes = max(len(np.unique(y)),2)

    ################################################
    # MODEL
    ################################################

    X_input = Input(shape=(n_features,))

    graph_conv = GraphConvolution(
        output_dim=n_hidden
    )(X_input)

    output_layer = Dense(
        units=n_classes,
        activation='softmax'
    )(graph_conv)

    model = Model(
        inputs=X_input,
        outputs=output_layer
    )

    ################################################
    # COMPILE
    ################################################


    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    ################################################
    # TRAIN
    ################################################

    model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=200,
        verbose=0
    )

    ################################################
    # PREDICT
    ################################################

    prob = model.predict(
        X_test,
        verbose=0
    )

    pred = np.argmax(
        prob,
        axis=1
    )

    ################################################
    # ATTACK SUCCESS RATE
    ################################################

    ATTACK, asr = attack(G)



    ################################################
    # METRICS
    ################################################

    acc = accuracy_score(
        y_test,
        pred
    )

    prec = precision_score(
        y_test,
        pred,
        average='macro',
        zero_division=0
    )

    ################################################
    # STORE
    ################################################

    ACC.append(acc)
    ASR.append(asr)
    PRE.append(prec)


