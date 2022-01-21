# Advanced Quantum Mechanics and Quantum Computing exam project


## Project
The main goal of this project is to apply Quantum Kernel Machine learing tecniques to data a data classification problem in High Energy Physics, description of the problem and the dataset can be found [here](https://qml-hep.github.io/qml_web/)

## Requirements
The project relies on the following python packges: 

- `qiskit`: IBM's quantum computing framework

- `matplotlib`, `seaoborn`: for plotting graphs

- `numpy` and `sklearn` : for the numerical heavy lifting and the machine learning algorithms and metrics used

- `a lot of patience` : training and testing this kind of algorithm is very very time consuming, the system on wich was tested is an `Intel Core i7 6700HQ - 8GB RAM`, running the entire test for 1 run takes more or less 1 hour and a half



## Repository description
The repository is contains 3 main components:

- `QSVM.ipynb` Experimental notebook used to test code

- `whole.py` Python File for running the entire project

- `utils.py` Utility file with functions for plotting and 

## How to run 

Pareser is not yet implemented, paramters are hard coded in the first lines of the python file.

**IMPORTANT**: change `data_path` variable. The file expects an `numpy` array which has the first k columns as features and the last one as label.

The imported array is then shuffled and divided in 80% for training and 20% for test.

Parameters:

- `n_c`            : number of features to consider for training and tesing models

- `encoders`       : encoders used to reduce data dimensionality

- `C_SVM_SAMPLES`  : samples used to train the classical model

- `Q_SVM_SAMPLES`  : samples used to train the quantum model

- `PREDICTIONS`    : samples used to test models

- `RUNS`           : numbers of run to perform

- `PROBA`          : control the fitting of svm to later retrive probability score and AUC




