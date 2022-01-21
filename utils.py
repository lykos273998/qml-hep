
from qiskit.circuit import QuantumCircuit, ParameterVector
import matplotlib.pyplot as plt
import numpy as np


def u2Reuploading(nqubits=8, nfeatures=16) -> QuantumCircuit:
    """
    Credits: Vasilis Belis (vbelis), bb511 (Patrick Odagiu) , GonzalesCastillo (Samuel Gonzales Castillo)

    https://github.com/QML-HEP/ae_qml

    Constructs the u2Reuploading feature map.
    @nqubits   :: Int number of qubits used.
    @nfeatures :: Number of variables in the dataset to be processed.
    returns :: The quantum circuit object form qiskit.
    """
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
        qc.u(
            np.pi / 2, x[feature], x[feature + 1], qubit
        )  # u2(φ,λ) = u(π/2,φ,λ)
    for i in range(nqubits):
        if i == nqubits - 1:
            break
        qc.cx(i, i + 1)
    for feature, qubit in zip(range(2 * nqubits, nfeatures, 2), range(nqubits)):
        qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)

    for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
        qc.u(x[feature], x[feature + 1], 0, qubit)

    return qc


def get_hist_proba(y_proba_train_cl, y_proba_test_cl, bg_train, bg_test, sig_train, sig_test, savepath = 'cmats/p_hist.png'):
    """
    Function that builds the histogram of the probabilities for which the model has predicted an observation to be signal or background
    arguments:

    @y_proba_train_cl :: probability scores for the ouput on training samples
    @y_proba_test_cl  :: probability scores for the ouput on test samples
    @bg_train         :: indices of samples belonging to background in the training set
    @bt_test          :: indices of samples belonging to background in the test set
    @sig_train        :: indices of samples belonging to signal in the training set
    @sig_test         :: indices of samples belonging to signal in the test set
    """
    bg_proba_train_cl  = y_proba_train_cl[bg_train] 
    sig_proba_train_cl = y_proba_train_cl[sig_train] 
    bg_proba_test_cl  = y_proba_test_cl[bg_test] 
    sig_proba_test_cl = y_proba_test_cl[sig_test] 
         
    #axs[0].sharey(axs[1])
    fig,axs = plt.subplots(1,2,sharey=False, tight_layout=True)
    fig.suptitle("") #Y

    #estetica
    axs[0].set_xlabel('p'),     axs[1].set_xlabel('p')
    axs[0].set_ylabel('count'), axs[1].set_ylabel('count')
    axs[0].set_xlim(0,1), axs[1].set_xlim(0,1)    
    #axs[0].set_yticks([]), axs[1].set_yticks([])
    axs[0].set_title('Training set')
    axs[0].hist(bg_proba_train_cl[:,0], alpha=0.6, label='background', color='black')
    axs[0].hist(sig_proba_train_cl[:,1], alpha=0.6, label='signal', color='green')
    axs[1].set_title('Test set')
    axs[1].hist(bg_proba_test_cl[:,0], alpha=0.6, label='background', color='black')
    axs[1].hist(sig_proba_test_cl[:,1], alpha=0.6, label='signal', color='green')
    axs[0].legend(loc='upper left'), axs[1].legend(loc='upper left')

    plt.savefig(savepath)

#defining feature list utility
feature_list = [
    'p_T Jet1',
    '\eta Jet1',
    '\phi Jet1',
    'E Jet1',
    'p_x Jet1',
    'p_y Jet1',
    'p_z Jet1',
    'b tag Jet1',
    'p_T Jet2',
    '\eta Jet2',      
    '\phi Jet2',    
    'E Jet2',
    'p_x Jet2',
    'p_y Jet2',
    'p_z Jet2',
    'b tag Jet2',
    'p_T Jet3',
    '\eta Jet3',
    '\phi Jet3',
    'E Jet3',
    'p_x Jet3',
    'p_y Jet3',
    'p_z Jet3', 
    'b tag Jet3',
    'p_T Jet4',
    '\eta Jet4',
    '\phi Jet4',
    'E Jet4',
    'p_x Jet4',
    'p_y Jet4',
    'p_z Jet4',
    'b tag Jet4',
    'p_T Jet5',
    '\eta Jet5',
    '\phi Jet5',
    'E Jet5',
    'p_x Jet5',
    'p_y Jet5',
    'p_z Jet5',
    'b tag Jet5',
    'p_T Jet6',
    '\eta Jet6',
    '\phi Jet6',
    'E Jet6',
    'p_x Jet6',
    'p_y Jet6',
    'p_z Jet6',
    'b tag Jet6',
    'p_T Jet7',
    '\eta Jet7',
    '\phi Jet7',
    'E Jet7',
    'p_x Jet7',
    'p_y Jet7',
    'p_z Jet7',
    'b tag Jet7',
    '\phi MET',
    'p_x MET',
    'p_y MET',
    'p_z MET',
    'p_T Lepton',
    '\eta Lepton',
    '\phi Lepton',
    'E Lepton',
    'p_x Lepton',
    'p_y Lepton',
    'p_z Lepton'
    
]
