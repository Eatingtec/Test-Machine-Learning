import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from UCI_datasets_loader import datasets_loader


datasets = {
    'abalone' : datasets_loader.load_abalone(),
    'breast_cancer_wisconsin' : datasets_loader.load_breast_cancer_wisconsin(),
    'ecoli' : datasets_loader.load_ecoli(),
    'glass' : datasets_loader.load_glass(),
    'hepatitis' : datasets_loader.load_hepatitis(),
    'iris' : datasets_loader.load_iris(),
    'lung_cancer' : datasets_loader.load_lung_cancer(),
    'transfusion' : datasets_loader.load_transfusion(),
    'winequality_red' : datasets_loader.load_winequality_red(),
    'winequality_white' : datasets_loader.load_winequality_white()
}

classifiers = {
    'decision_tree' : DecisionTreeClassifier(criterion="gini",min_impurity_split=1e-7),
    'svm' : SVC(kernel='rbf',C=1.0),
    'neural_network' : MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1,max_iter=200)
}
result = {}

for dataset_name in datasets.keys():
    dataset = datasets[dataset_name]
    for classifier_name in classifiers.keys():
        classifier = classifiers[classifier_name]
        scores = cross_val_score(classifier,dataset.data,dataset.target,cv=10,scoring='accuracy')
        if dataset_name not in result.keys():
            result[dataset_name] = {}
        result[dataset_name][classifier_name] = scores.mean()


print result

import json

with open('result.json', 'w') as file:
    json.dump(result,file)
