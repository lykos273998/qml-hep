import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
classic_cmap = plt.get_cmap('Reds')

from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA, KernelPCA

from qiskit import BasicAer, Aer
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel


from sklearn.kernel_approximation import Nystroem
from sklearn.utils import shuffle

from utils import feature_list, get_hist_proba, u2Reuploading

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

import sys

"""
Final project for Quantum Computing course @Units

Authors: Francesco Tomba, Jasmin Bougammoura &  Giacomo Zelbi
January 2022

Testing Quantum Kernel SVM approach to binary classification problem in HEP

Details on the dataset: https://qml-hep.github.io/qml_web/


/!\ this code is higly cpu demanding on xy and zz maps
/!\ for a quick test use only u2 map
"""


"""
Parameters for the test:

@n_c            :: number of features to consider for training and tesing models
@encoders       :: encoders used to reduce data dimensionality 
@C_SVM_SAMPLES  :: samples used to train the classical model
@Q_SVM_SAMPLES  :: samples used to train the quantum model
@PREDICTIONS    :: samples used to test models
@RUNS           :: numbers of run to perform
@PROBA          :: control the fitting of svm to later retrive probability score and AUC

"""
n_c = 8
encoders = ['auc','pca','nys']
#encoders = ['auc']

#modify this line to use one particular map
f_maps = ['u2','zz','xy']
#f_maps = ['u2']
C_SVM_SAMPLES = 500
Q_SVM_SAMPLES = 500
PREDICTIONS = 500
RUNS = 3
PROBA = True

"""

Selection of the first 16 features based on auc.
AUC is precomputed on : https://qml-hep.github.io/qml_web/

"""

FEATURES = [32, 24, 40, 16, 8, 0, 48, 3, 51, 11, 19, 43, 27, 35, 57, 47]



"""
Change this variable to change data loading path
"""
data_path = '../data/'

data = np.load(f"{data_path}x_data_normalized.npy")


from time import time
#path to save confusion matrix and other graphs
CMAT_PATH = 'cmats/'

seed = 12345

"""
Getting the feature maps and a backend for evaluating later
"""

zz = ZZFeatureMap(feature_dimension=n_c, reps=2, entanglement='linear')
u2 = u2Reuploading(nqubits = n_c//2, nfeatures=n_c)
xy = PauliFeatureMap(feature_dimension=n_c, reps=2, paulis=['XY'],entanglement = 'linear')

backend = QuantumInstance(
    BasicAer.get_backend("statevector_simulator"), shots=1024, seed_simulator=seed, seed_transpiler=seed
)

print(f"running experiment with: \n\t classical svm samples {C_SVM_SAMPLES} \n\t Quantum SVM samples {Q_SVM_SAMPLES}")
print("")

measure_log = []
for run in range(RUNS):
    tr_begin = time()

    """
    shuffling data and getting training and test set for each run
    """
    data = shuffle(data)

    x_train = data[:int(0.8*len(data)),:-1]
    y_train = data[:int(0.8*len(data)),-1]

    x_test = data[int(0.8*len(data)):,:-1]

    y_test = data[int(0.8*len(data)):,-1]

    bg_train = np.where(y_train[:PREDICTIONS] == 0)
    sig_train = np.where(y_train[:PREDICTIONS] == 1)
    bg_test = np.where(y_test[:PREDICTIONS] == 0)
    sig_test = np.where(y_test[:PREDICTIONS] == 1)

    """
    Select an encoder
    """

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

        """

        Declaring svm object
        Using rbf function

        """

        svm = SVC(kernel = 'rbf', probability=PROBA)

        """
        Log data
        """
        row_log = [f"{encoder}+rbf"]

        print(f"running classical svm w. encoder {encoder}")

        """
        Fit the model and take the time
        """
        begin = time()
        svm.fit(x_tf_train[:C_SVM_SAMPLES], y_train[:C_SVM_SAMPLES])
        end = time()
        row_log.append(end-begin)
        print(f"\t training time: {end - begin : .2f}")

        #predictions and confusion matrix
        """
        Use the model to predict labels of data and take the time
        """
        begin = time()
        y_pred_train = svm.predict(x_tf_train[:PREDICTIONS])
        end = time()
        row_log.append(end-begin)
        print(f"\t prediction time training set: {end - begin : .2f}")


        #plot the confusion matrix
        cm = confusion_matrix(y_train[:PREDICTIONS],y_pred_train, normalize='true')
        sns.heatmap(cm,vmin=0,vmax=1,annot = True, cmap=classic_cmap)
        plt.xlabel('Predicted label'), plt.ylabel('True label')
        plt.title(f"classical svm w. encoder {encoder} training set")
        plt.savefig(f"{CMAT_PATH}classical_{encoder}_train.jpg")

        """
        Use the model to predict labels of data and take the time
        """
        begin = time()
        y_pred_test = svm.predict(x_tf_test[:PREDICTIONS])
        end = time()
        row_log.append(end-begin)
        print(f"\t prediction time test set: {end - begin : .2f}")

        cm = confusion_matrix(y_test[:PREDICTIONS],y_pred_test, normalize='true')
        sns.heatmap(cm,vmin=0,vmax=1,annot = True, cmap=classic_cmap)
         
        plt.xlabel('Predicted label'), plt.ylabel('True label')
        plt.title(f"classical svm w. encoder {encoder} test set")
        plt.savefig(f"{CMAT_PATH}classical_{encoder}_test.jpg")

        if PROBA:
            """
            If proba is true then

            Calculate probability of each example belonging to one or other class
            The use this probability estimate to calculate roc auc score

            Then plot the distribution of examples over probabilities
            the goal is to check if the majority of examples belonging to signal has
            prob to belonging to signal approaching 1

            """
            print(f"\t\t CALCULATING ROC AUC SCORES: this may take some time")
            y_proba_train = svm.predict_proba(x_tf_train[:PREDICTIONS])
            score = roc_auc_score(y_train[:PREDICTIONS], y_proba_train[:,1])

            print(f"\t\t Classical svm ROC AUC score TRAINING SET {score :.2f}")
            row_log.append(score)
            y_proba_test = svm.predict_proba(x_tf_test[:PREDICTIONS])
            score = roc_auc_score(y_test[:PREDICTIONS], y_proba_test[:,1])
            row_log.append(score)
            print(f"\t\t Classical svm ROC AUC score TEST SET {score :.2f}")

            """
            Distribuzioni di probabilità
            """
            get_hist_proba(y_proba_train, y_proba_test, bg_train, bg_test, sig_train, sig_test, savepath = f"{CMAT_PATH}p_hist_{encoder}_rbf_hist.png")

        measure_log.append(row_log)


        """
        Quantum computing part
        """
        for feature_map in f_maps:

            if feature_map == 'u2':
                kernel = QuantumKernel(feature_map=u2 , quantum_instance=backend)
            if feature_map == 'zz':
                kernel = QuantumKernel(feature_map=zz , quantum_instance=backend)
            if feature_map == 'xy':
                kernel = QuantumKernel(feature_map=xy , quantum_instance=backend)


            qsvm = SVC(kernel=kernel.evaluate, probability=PROBA)
            row_log = [f"{encoder}+{feature_map}"]
            print(f"running quantum svm w. encoder {encoder.upper()} feature map {feature_map.upper()}")

            begin = time()
            qsvm.fit(x_tf_train[:Q_SVM_SAMPLES], y_train[:Q_SVM_SAMPLES])
            end = time()
            row_log.append(end-begin)
            print(f"\t training time: {end - begin : .2f}")

            begin = time()
            y_pred_train = qsvm.predict(x_tf_train[:PREDICTIONS])
            end = time()
            row_log.append(end-begin)
            print(f"\t prediction time training set: {end - begin : .2f}")

            cm = confusion_matrix(y_train[:PREDICTIONS],y_pred_train, normalize='true')
            sns.heatmap(cm,vmin=0,vmax=1,annot = True, cmap=classic_cmap)
             
            plt.xlabel('Predicted label'), plt.ylabel('True label')
            plt.title(f"quantum svm w. encoder {encoder} map {feature_map} training set ")
            plt.savefig(f"{CMAT_PATH}quantum_{encoder}_{feature_map}_train.jpg")
            
            begin = time()
            y_pred_test = qsvm.predict(x_tf_test[:PREDICTIONS])
            end = time()
            row_log.append(end-begin)
            print(f"\t prediction time test set: {end - begin : .2f}")

            cm = confusion_matrix(y_test[:PREDICTIONS],y_pred_test, normalize='true')
            sns.heatmap(cm,vmin=0,vmax=1,annot = True, cmap=classic_cmap)
             
            plt.xlabel('Predicted label'), plt.ylabel('True label')
            plt.title(f"quantum svm w. encoder {encoder} map {feature_map}  test set")
            plt.savefig(f"{CMAT_PATH}quantum_{encoder}_{feature_map}_test.jpg")

            if PROBA:
                print(f"\t\t CALCULATING ROC AUC SCORES: this may take some time")
                y_proba_train = qsvm.predict_proba(x_tf_train[:PREDICTIONS])
                score = roc_auc_score(y_train[:PREDICTIONS], y_proba_train[:,1])

                print(f"\t\t Quantum svm ROC AUC score TRAINING SET {score :.2f}")
                row_log.append(score)

                y_proba_test = qsvm.predict_proba(x_tf_test[:PREDICTIONS])
                score = roc_auc_score(y_test[:PREDICTIONS], y_proba_test[:,1])
                row_log.append(score)

                print(f"\t\t Quantum svm ROC AUC score TEST SET {score :.2f}")

                """
                Distribuzioni di probabilità 
                """
                get_hist_proba(y_proba_train, y_proba_test, bg_train, bg_test, sig_train, sig_test, savepath = f"{CMAT_PATH}p_hist_{encoder}_{feature_map}_hist.png")        

            measure_log.append(row_log)
    tr_end = time()
    print(f"run time {tr_end - tr_begin : .2f}")
print("\n\n")
print(measure_log)
        
import pickle
from datetime import datetime

with open(f"measure_log_{datetime.now()}", 'wb') as fp:
    pickle.dump(measure_log, fp)



