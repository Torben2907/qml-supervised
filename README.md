# A quantum-based approach for supervised learning (bachelor thesis) 

This repository provides code for comparing classical machine learning models with their quantum-hybrid counterparts. We've trained and evaluated these models on real-world biomedical data to investigate whether the quantum-hybrid versions can achieve higher accuracy than their classical counterparts, potentially demonstrating a quantum advantage.

## Welcome to QMLab! 🧪

The package created is called `QMLab` - it's written entirely in Python, using [Qiskit](https://github.com/Qiskit/qiskit) for the quantum mechanical backend and [Sklearn](https://github.com/scikit-learn/scikit-learn) for the classical backend. The main class we use is [`QSVC`](https://github.com/Torben2907/qml-supervised/blob/master/src/qmlab/qsvm.py#L12) - short for **Q**uantum **S**upport **V**ector **M**achine. It inherits from the [`SVC`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/_classes.py#L604) class of Scikit-Learn and extends its functionality by allowing quantum kernel as well as classical ones.

[Get started with the Jupyter-Tutorial-Series here!](/tutorials/classical_learning.ipynb)

## Information about the data 🧬

We're dealing with 9 different biomedical datasets in this thesis. The collection is coming from <cite>Jacqueline Beinecke & Dominik Heider [[1]]</cite>. All datasets contain an imbalance of class labels and cannot be seperated linearly.

![Overview of the datasets](/figures/information_datasets.png)

## What is a Quantum Support Vector Machine? 🤔

Since real quantum hardware is noisy and prone for errors a whole field submerged of quantum hybrid algorithms 
which are based on classical learning algorithms but outsource suitable parts to a quantum computer.

The basic support vector machine algorithm can only classify data that is linearly separable. 
It becomes way more powerful when we introduce a class of functions called **kernels**. 
Kernels allow us to map our data in a higher, perhaps infinite dimensional space where it is linearly separable. 



## Can we train a Quantum Kernel? 🤨

---
[1]: https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00283-6#Tab1