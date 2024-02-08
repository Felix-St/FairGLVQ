import numpy as np

import argparse

from aif360.sklearn.metrics import statistical_parity_difference
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold
import utilities.distributions as distributions
from sklearn.preprocessing import MinMaxScaler

from models.Auxiliary_models import DummyClassifier
from utilities.INP import get_debiasing_projection
from models.GLVQ_models import Fair_GLVQ, GLVQ
from sklearn.linear_model import SGDClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Local',
                    help="(Single) Dataset which should be benchmarked", choices=['Local', 'XOR'])
parser.add_argument('--epochs', type=int, help="Number of epochs", default=250)
parser.add_argument('--folds', type=int, help="Number of cross validation folds", default=5)

args = parser.parse_args()

epochs = args.epochs
num_folds = args.folds
dataset = args.dataset

sd_generated = 4906313 # random.org

np.random.seed(sd_generated)

if dataset == "XOR":
    X = distributions.create_synthetic_dataset_XOR(800, seed=sd_generated)
    num_prots = 4
    reg = 1.25
elif dataset == "Local":
    X = distributions.create_synthetic_dataset_Local(800, 2, seed=sd_generated)
    num_prots = 5
    reg = 1.5
else:
    raise Exception("Unknown Dataset")

seed_generator = np.random.SeedSequence(entropy=sd_generated)
seeds = seed_generator.generate_state(num_folds)

kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=sd_generated)

glvq_acc = []
fglvq_acc = []
inp_acc = []

glvq_stp = []
fglvq_stp = []
inp_stp = []

dummy_accs = []
dummy_stps = []


for i, (train_index, test_index) in enumerate(kf.split(X[:,0:3], X[:,3])):
    X_train = X[train_index,0:3]
    y_train = X[train_index,3]

    X_test = X[test_index,0:3]
    y_test = X[test_index,3]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #### Dummy
    dummy = DummyClassifier()
    dummy.fit(np.array(y_train,dtype='int64'))
    dummy_pred = dummy.predict(np.transpose(X_test))

    dummy_accuracy = accuracy_score(dummy_pred, y_test)
    dummy_statistical_parity = abs(statistical_parity_difference(y_test, np.asarray(dummy_pred), prot_attr=X_test[:, 2]))

    dummy_accs.append(dummy_accuracy)
    dummy_stps.append(dummy_statistical_parity)

   
    ### GLVQ
    classifier_glvq = GLVQ(X_train[:,0:2], y_train, num_prots, activation="swish", beta=2.0, lr=0.005, initializer="KMEANS", seed=seeds[i])
    classifier_glvq.fit(epochs=epochs, batch_size=250, verbose=50)

    glvq_pred = classifier_glvq.predict(X_test[:,0:2])

    accuracy = accuracy_score(glvq_pred, y_test)
    statistical_parity = abs(statistical_parity_difference(y_test, np.asarray(glvq_pred), prot_attr=X_test[:,2]))

    glvq_acc.append(accuracy)
    glvq_stp.append(statistical_parity)

    ### FairGLVQ
    classifier_fglvq = Fair_GLVQ(X_train, y_train, num_prots , 2, activation="swish", beta=2.0, lr=0.005, regularization=reg, initializer="FAIR_KMEANS",use_protected_attribute=False, seed=seeds[i])
    classifier_fglvq.fit(epochs=epochs, batch_size=250,verbose=50)

    fglvq_pred = classifier_fglvq.predict(X_test[:,0:2])

    accuracy = accuracy_score(fglvq_pred, y_test)
    statistical_parity = abs(statistical_parity_difference(y_test, np.asarray(fglvq_pred), prot_attr=X_test[:,2]))

    fglvq_acc.append(accuracy)
    fglvq_stp.append(statistical_parity)

    #### GLVQ + INP
    classifier_class = SGDClassifier

    P, _, _ = get_debiasing_projection(classifier_class, {'random_state':seeds[i]}, 1, 2,
                                       True, 0.0, X_train[:,0:2],
                                       X_train[:,2], X_test[:,0:2], X_test[:,2],
                                       by_class=False)


    scaler2 = MinMaxScaler()
    X_proj = scaler2.fit_transform(X_train[:,0:2]@P)
    X_proj_test = scaler2.transform(X_test[:,0:2]@P)

    classifier_glvq_inp = GLVQ(X_proj[:, 0:2], y_train, num_prots, activation="swish", beta=2.0, lr=0.005, initializer="KMEANS", seed=seeds[i])
    classifier_glvq_inp.fit(epochs=epochs, batch_size=250, verbose=50)

    inp_pred = classifier_glvq_inp.predict(X_proj_test)

    accuracy = accuracy_score(inp_pred, y_test)
    statistical_parity = abs(statistical_parity_difference(y_test, np.asarray(inp_pred), prot_attr=X_test[:, 2]))

    inp_acc.append(accuracy)
    inp_stp.append(statistical_parity)

    """ Uncomment if you are interested in comparing the results of GLVQ and FairGLVQ visually
    from utilities.vis import visualize_result, visualize_result_and_prediction
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    visualize_result_and_prediction(axs[0], X_test[:, 0:2], X_test[:, 2], glvq_pred, classifier_glvq, "Prediction GLVQ")
    visualize_result_and_prediction(axs[1], X_test[:, 0:2], X_test[:, 2], fglvq_pred, classifier_fglvq,"Prediction FGLVQ")
    plt.show()
    """

print("\\begin{table*}[h]")
print("\\begin{center}")
print("\\label{tb:synthetic_results}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{l@{\:\:}lcccc}")
print("\\toprule")
print(" && $\\underset{\\text{(ours)}}{\\text{FairGLVQ}}$ & INP & GLVQ & const.  \\\\")
print("\\midrule \\multirow{2}{*}{\\rotatebox[origin=c]{90}{" + dataset + "}}")
text_accuracies = "& Acc &${fglvq:.2f}(\\pm {fglvqstd:.2f})$ & ${INP:.2f} (\\pm {INPstd:.2f})$ & ${glvq:.2f}(\\pm {glvqstd:.2f})$ & ${Dummy:.2f}(\\pm {Dummystd:.2f})$\\\\"
print(text_accuracies.format(fglvq=np.mean(fglvq_acc), fglvqstd=np.std(fglvq_acc), INP=np.mean(inp_acc),INPstd=np.std(inp_acc),glvq=np.mean(glvq_acc),glvqstd=np.std(glvq_acc),Dummy=np.mean(dummy_accs),Dummystd=np.std(dummy_accs)))
text_fairness = "& SP &${fglvq:.2f}(\\pm {fglvqstd:.2f})$ & ${INP:.2f} (\\pm {INPstd:.2f})$ & ${glvq:.2f}(\\pm {glvqstd:.2f})$ &  ${Dummy:.2f}(\\pm {Dummystd:.2f})$\\\\"
print(text_fairness.format(fglvq=np.mean(fglvq_stp), fglvqstd=np.std(fglvq_stp), INP=np.mean(inp_stp),INPstd=np.std(inp_stp),glvq=np.mean(glvq_stp),glvqstd=np.std(glvq_stp),Dummy=np.mean(dummy_stps),Dummystd=np.std(dummy_stps)))
print("\\bottomrule")
print("\\end{tabular}}")
print("\\end{center}")
print("\\end{table*}")
