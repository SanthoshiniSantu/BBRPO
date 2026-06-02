import GNN.gnn
import GNN_JSO.gnn
import GNN_WWPA.gnn
import Proposed_JWG_GUARD.gnn
import Run_CIFAR
import Run_MNIST
import Run_MUTAG
def callmain(ds,tr):
    if ds == 'CIFAR':
        X,Y, G = Run_CIFAR.main()
    elif ds == 'MNIST':
        X,Y, G = Run_MNIST.main()
    else:
        X,Y, G = Run_MUTAG.main()

    ACC, ASR, PRE =[],[], []


    GNN.gnn.classify(X,Y,tr, G,ACC, ASR, PRE)
    GNN_JSO.gnn.classify(X,Y,tr, G,ACC, ASR, PRE)
    GNN_WWPA.gnn.classify(X,Y,tr, G,ACC, ASR, PRE)
    Proposed_JWG_GUARD.gnn.classify(X,Y,tr, G,ACC, ASR, PRE)

    return ACC, ASR, PRE

if __name__ == "__main__":

    print("Available Datasets:")
    print("1. CIFAR")
    print("2. MNIST")
    print("3. MUTAG")

    choice = int(input("\nEnter Dataset (1-CIFAR, 2-MNIST, 3-MUTAG): "))

    if choice == 1:
        ds = "CIFAR"
    elif choice == 2:
        ds = "MNIST"
    elif choice == 3:
        ds = "MUTAG"
    else:
        raise ValueError("Invalid Dataset Choice")

    tr = float(input("Enter Training Percentage (50/60/70/80/90): "))

    tr = tr / 100.0

    ACC, ASR, PRE = callmain(ds, tr)

    print("\nTraining Ratio :", tr)
    print("Accuracy :", ACC)
    print("ASR :", ASR)
    print("Precision :", PRE)