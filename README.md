# Anomaly-Detection-in-Cryptocurrency-Transactions-with-Active-Learning


#Overview

This repository includes the necessary code to replicate the results presented in the dissertation titled "Anomaly Detection in Cryptocurrency Transactions with Active Learning." The provided code base covers both the Supervised baseline and the Active Learning experiments.

#Requirements

Please ensure that you have the following main requirements to proceed with the task:

Python 3.8.10 or a compatible version.
The following Python packages:
- networkx
- sklearn
- matplotlib
- seaborn
- pyod
- xgboost
- scipy
- numpy
- pandas

Once you have installed the necessary packages, please download the Elliptic Bitcoin dataset from the following link: https://www.kaggle.com/ellipticco/elliptic-data-set. 
After downloading the dataset, save the three .csv files (elliptic_txs_features.csv, elliptic_txs_classes.csv, and elliptic_txs_edgelist.csv) in the data/ directory.

#Experiments

To reproduce the results, you can find all the necessary Python scripts in the src/Experiments and src/Functions directories. To run a specific experiment, navigate to the project's root folder in the terminal and execute the corresponding Python script:

- For the supervised methods: src/Experiments/Supervised_Baseline.py
- For the active learning experiments:
  - Baseline: src/Experiments/AL_Baseline.py
  - Scenario 1: src/Experiments/AL_Scenario1.py
  - Scenario 2: src/Experiments/AL_Scenario2.py
- For t-SNE projection of the predicted labels from the supervised methods on the test set: src/Experiments/t-SNE_Projections.py


#Author

- Leandro Cunha: cunhaleandro89@gmail.com


