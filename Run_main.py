from sklearn.model_selection import KFold
import numpy as np

import Proposed_JWG_GUARD.gnn
import Run_CIFAR
import Run_MNIST
import Run_MUTAG


def load_dataset(ds):

    if ds == "CIFAR":
        return Run_CIFAR.main()

    elif ds == "MNIST":
        return Run_MNIST.main()

    elif ds == "MUTAG":
        return Run_MUTAG.main()

    else:
        raise ValueError("Invalid Dataset")


def callmain(ds,tr,kfold):

    X,Y,G = load_dataset(ds)

    kf = KFold(
        n_splits=kfold,
        shuffle=True,
        random_state=42
    )

    ACC=[]
    ASR=[]
    PRE=[]

    for fold,(train_idx,test_idx) in enumerate(kf.split(X),1):

        print(f"\nFold {fold}/{kfold}")

        X_train=X[train_idx]
        Y_train=Y[train_idx]

        X_test=X[test_idx]
        Y_test=Y[test_idx]

        Proposed_JWG_GUARD.gnn.classify(
            X_train,
            Y_train,
            X_test,
            Y_test,
            tr,
            G,
            ACC,
            ASR,
            PRE
        )

    return np.mean(ACC),np.mean(ASR),np.mean(PRE)


if __name__=="__main__":

    print("Available Datasets")
    print("1. CIFAR")
    print("2. MNIST")
    print("3. MUTAG")

    choice=int(input("\nEnter Dataset (1-3): "))

    if choice==1:
        ds="CIFAR"
    elif choice==2:
        ds="MNIST"
    elif choice==3:
        ds="MUTAG"
    else:
        raise ValueError("Invalid Dataset")

    tr=float(input("Enter Training Percentage (50/60/70/80/90): "))
    tr=tr/100.0

    kfold=int(input("Enter K-Fold Value (5/6/7/8/9): "))

    ACC,ASR,PRE=callmain(ds,tr,kfold)

    print("\n==========================")
    print("Dataset :",ds)
    print("Training Ratio :",tr)
    print("K-Fold :",kfold)
    print("==========================")
    print("Accuracy :",ACC)
    print("ASR :",ASR)
    print("Precision :",PRE)

#-------------- K Fold analysis---------
import numpy as np
from sklearn.model_selection import KFold

import Proposed_JWG_GUARD.gnn
import Run_CIFAR
import Run_MNIST
import Run_MUTAG


def load_dataset(ds):

    if ds=='CIFAR':
        return Run_CIFAR.main()

    elif ds=='MNIST':
        return Run_MNIST.main()

    elif ds=='MUTAG':
        return Run_MUTAG.main()

    else:
        raise ValueError("Invalid Dataset")


def callmain(ds,kfold):

    X,Y,G=load_dataset(ds)

    X=np.array(X)
    Y=np.array(Y)

    ACC=[]
    ASR=[]
    PRE=[]

    kf=KFold(n_splits=kfold,shuffle=True,random_state=42)

    for fold,(train_idx,test_idx) in enumerate(kf.split(X),1):

        print("Running Fold :",fold)

        X_train=X[train_idx]
        Y_train=Y[train_idx]

        X_test=X[test_idx]
        Y_test=Y[test_idx]

        Proposed_JWG_GUARD.gnn.classify_kf(
            X_train,
            Y_train,
            X_test,
            Y_test,
            G,
            ACC,
            ASR,
            PRE
        )

    return np.mean(ACC),np.mean(ASR),np.mean(PRE)


if __name__=="__main__":

    print("1. CIFAR")
    print("2. MNIST")
    print("3. MUTAG")

    choice=int(input("Enter Dataset : "))

    if choice==1:
        ds='CIFAR'
    elif choice==2:
        ds='MNIST'
    elif choice==3:
        ds='MUTAG'
    else:
        raise ValueError("Invalid Dataset")

    kfold=int(input("Enter K Fold Value (5/6/7/8/9): "))

    ACC,ASR,PRE=callmain(ds,kfold)

    print("\n====================")
    print("Dataset   :",ds)
    print("K Fold    :",kfold)
    print("Accuracy  :",ACC)
    print("ASR       :",ASR)
    print("Precision :",PRE)
    print("====================")