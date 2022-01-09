import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA, KernelPCA

from qiskit import BasicAer, Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data

from sklearn.kernel_approximation import Nystroem
from sklearn.utils import shuffle

from utils import feature_list, u2Reuploading

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

import sys

n_c = 16
C_SVM_SAMPLES = 10000
Q_SVM_SAMPLES = 500
PREDICTIONS = 500



FEATURES = [32, 24, 40, 16, 8, 0, 48, 3, 51, 11, 19, 43, 27, 35, 57, 47]

PROBA = True

def gen_hist(bg,sig,feature):
    
    hist_bg = np.histogram(bg[:,feature],bins = 30, density=True)
    hist_sig = np.histogram(sig[:,feature],bins = 30, density=True)

    plt.figure(figsize=(6,6), dpi=(100))
    plt.step(hist_bg[1][:-1],hist_bg[0], color = tuple(np.array([8,41,100,255])/ 255.))
    plt.step(hist_sig[1][:-1],hist_sig[0], color=tuple(np.array([200,20,8,255])/ 255.))

    plt.fill_between(hist_bg[1][:-1],hist_bg[0], np.zeros_like(hist_bg[0]), 
                    color = tuple(np.array([8,41,100,100])/ 255.), step = 'pre', label = 'Background')
    plt.fill_between(hist_sig[1][:-1],hist_sig[0], np.zeros_like(hist_sig[0]), 
                    color = tuple(np.array([200,20,8,50])/ 255.), step = 'pre', label = 'Signal')
    plt.legend()
    plt.ylabel('Density')
    plt.xlabel(feature_list[feature])


data_path = '../data/'

data = np.load(f"{data_path}x_data_normalized.npy")
data = shuffle(data)

x_train = data[:int(0.8*len(data)),:-1]
y_train = data[:int(0.8*len(data)),-1]

x_test = data[int(0.2*len(data)):,:-1]

y_test = data[int(0.2*len(data)):,-1]

bg_train = np.where(y_train == 0)
sig_train = np.where(y_train == 1)


encoders = ['auc','pca','nys']
#encoders = ['auc']

from time import time
CMAT_PATH = 'cmats/'

seed = 12345


#f_maps = ['u2','zz']
f_maps = ['zz']

zz = ZZFeatureMap(feature_dimension=n_c, reps=2, entanglement='linear')
u2 = u2Reuploading(nqubits = n_c//2, nfeatures=n_c)

backend = QuantumInstance(
    BasicAer.get_backend("statevector_simulator"), shots=1024, seed_simulator=seed, seed_transpiler=seed
)


print(f"running experiment with: \n\t classical svm samples {C_SVM_SAMPLES} \n\t Quantum SVM samples {Q_SVM_SAMPLES}")
print("")

for encoder in encoders:
    if encoder == 'pca':
        
        transformer = PCA(n_components= n_c)
        transformer.fit(x_train)

        x_tf_train = transformer.transform(x_train)
        x_tf_test = transformer.transform(x_test)

    if encoder == 'nys':
        
        transformer = Nystroem(kernel='laplacian',gamma=0.2,n_components=n_c)
        transformer.fit(x_train,y_train)

        x_tf_train = transformer.transform(x_train)
        x_tf_test = transformer.transform(x_test)
    
    if encoder == 'auc':
        x_tf_train = x_train[:,FEATURES]
        x_tf_test = x_test[:,FEATURES]

        x_tf_train = x_tf_train[:,:n_c]
        x_tf_test = x_tf_test[:,:n_c]

    #declaring svm object

    svm = SVC(kernel = 'rbf', probability=PROBA)


    print(f"running classical svm w. encoder {encoder}")
    begin = time()
    svm.fit(x_tf_train[:C_SVM_SAMPLES], y_train[:C_SVM_SAMPLES])
    end = time()

    print(f"\t training time: {end - begin : .2f}")

    #predictions and confusion matrix
    begin = time()
    y_pred = svm.predict(x_tf_train[:PREDICTIONS])
    end = time()

    print(f"\t prediction time training set: {end - begin : .2f}")

    cm = confusion_matrix(y_train[:PREDICTIONS],y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"classical svm w. encoder {encoder} training set")
    plt.savefig(f"{CMAT_PATH}classical_{encoder}_train.jpg")


    begin = time()
    y_pred = svm.predict(x_tf_test[:PREDICTIONS])
    end = time()

    print(f"\t prediction time test set: {end - begin : .2f}")

    cm = confusion_matrix(y_test[:PREDICTIONS],y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"classical svm w. encoder {encoder} test set")
    plt.savefig(f"{CMAT_PATH}classical_{encoder}_test.jpg")

    if PROBA:
        print(f"\t\t CALCULATING ROC AUC SCORES: this may take some time")
        y_proba = svm.predict_proba(x_tf_train[:PREDICTIONS])
        score = roc_auc_score(y_train[:PREDICTIONS], y_proba[:,0])

        print(f"\t\t Classical svm ROC AUC score TRAINING SET {max(score,1 - score) :.2f}")

        y_proba = svm.predict_proba(x_tf_test[:PREDICTIONS])
        score = roc_auc_score(y_test[:PREDICTIONS], y_proba[:,0])

        print(f"\t\t Classical svm ROC AUC score TEST SET {max(score,1 - score):.2f}")



    """
    Quantum computing part
    """
    for feature_map in f_maps:

        if feature_map == 'u2':
            kernel = QuantumKernel(feature_map=u2 , quantum_instance=backend)
        if feature_map == 'zz':
            kernel = QuantumKernel(feature_map=zz , quantum_instance=backend)


        qsvm = SVC(kernel=kernel.evaluate, probability=PROBA)

        print(f"running quantum svm w. encoder {encoder.upper()} feature map {feature_map.upper()}")

        begin = time()
        qsvm.fit(x_tf_train[:Q_SVM_SAMPLES], y_train[:Q_SVM_SAMPLES])
        end = time()

        print(f"\t training time: {end - begin : .2f}")

        begin = time()
        y_pred = qsvm.predict(x_tf_train[:PREDICTIONS])
        end = time()

        print(f"\t prediction time training set: {end - begin : .2f}")

        cm = confusion_matrix(y_train[:PREDICTIONS],y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"quantum svm w. encoder {encoder} map {feature_map} training set ")
        plt.savefig(f"{CMAT_PATH}quantum_{encoder}_{feature_map}_train.jpg")
        
        begin = time()
        y_pred = qsvm.predict(x_tf_test[:PREDICTIONS])
        end = time()

        print(f"\t prediction time test set: {end - begin : .2f}")

        cm = confusion_matrix(y_test[:PREDICTIONS],y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"quantum svm w. encoder {encoder} map {feature_map}  test set")
        plt.savefig(f"{CMAT_PATH}quantum_{encoder}_{feature_map}_test.jpg")

        if PROBA:
            print(f"\t\t CALCULATING ROC AUC SCORES: this may take some time")
            y_proba = qsvm.predict_proba(x_tf_train[:PREDICTIONS])
            score = roc_auc_score(y_train[:PREDICTIONS], y_proba[:,0])

            print(f"\t\t Quantum svm ROC AUC score TRAINING SET {max(score,1 - score) :.2f}")

            y_proba = qsvm.predict_proba(x_tf_test[:PREDICTIONS])
            score = roc_auc_score(y_test[:PREDICTIONS], y_proba[:,0])

            print(f"\t\t Quantum svm ROC AUC score TEST SET {max(score,1 - score) :.2f}")



    




