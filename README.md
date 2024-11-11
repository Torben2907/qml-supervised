# A quantum-based approach for supervised learning (bachelor thesis) 

This repository provides the code for comparing classical machine learning models with their quantum-hybrid based versions.

This repository provides code for comparing classical machine learning models with their quantum-hybrid counterparts. We've trained and evaluated these models on real-world biomedical data to investigate whether the quantum-hybrid versions can achieve higher accuracy than their classical counterparts, potentially demonstrating a quantum advantage.

## Welcome to QMLab! 🧪

The package created is called `QMLab` - it's written entirely in Python, using [https://github.com/Qiskit/qiskit](Qiskit) for the quantum mechanical backend and [https://github.com/scikit-learn/scikit-learn](Scikit-Learn) for the classical backend. The main class used is `QSVC` - short for Quantum Support Vector Machine. It inherits from the SVC class of Scikit-Learn and extends its functionality by providing quantum kernel instead of classical ones.

[Get started with the Jupyter-Tutorial-Series here!](./tutorials/classical_learning.ipynb)

## Information about the biomedical data 🧬

We're dealing with 9 different datasets in this thesis. The collection is coming from <cite>Jacqueline Beinecke & Dominik Heider[1]</cite>. All datasets contain an imbalance of class labels and cannot be seperated linearly.

![Overview of the datasets](/figures/information_datasets.png)

## What is a Quantum Support Vector Machine? 🤔

## Can we train a Quantum Kernel? 🤨

---
[1]: https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00283-6#Tab1