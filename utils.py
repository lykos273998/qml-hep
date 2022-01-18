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


from calendar import c
from locale import normalize
from qiskit.circuit import QuantumCircuit, ParameterVector
import matplotlib.pyplot as plt
import numpy as np


def u2Reuploading(nqubits=8, nfeatures=16) -> QuantumCircuit:
    """
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


def generate_hist_prob(y_true, y_model, y_proba, title, path):
    n_class = np.shape(y_proba)[1]
    n = np.shape(y_proba)[0]
    for c_true in range(n_class):
        
        idx = np.where(y_true == c_true)
        plt.figure(figsize=(5,5))

        print(np.shape(y_proba[idx, c_true]))
        plt.hist(y_proba[idx, c_true].T, alpha = 0.8)
        plt.title(f"{title} class {c_true}")
        plt.savefig(f"{path}_c{c_true}.jpg")

