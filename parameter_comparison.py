import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from UCI_datasets_loader import datasets_loader
import matplotlib.pyplot as plt

datasets = [
            datasets_loader.load_ecoli(),
            datasets_loader.load_winequality_red(),
            datasets_loader.load_transfusion()
        ]
colors = ['r','g','b']
labels = ['ecoli','winequality_red','transfusion']

#for Decision Tree
def compare_different_min_samples_leaf():
    i = 0
    for dataset in datasets:
        min_samples_leaf_numbers = range(1,21)
        min_samples_leaf_number_scores = []
        for min_samples_leaf_number in min_samples_leaf_numbers:
            classifier = DecisionTreeClassifier(criterion="gini",min_impurity_split=1e-7,min_samples_leaf=min_samples_leaf_number)
            scores = cross_val_score(classifier,dataset.data,dataset.target,cv=10,scoring='accuracy')
            min_samples_leaf_number_scores.append(scores.mean())
        my_print(min_samples_leaf_number_scores)
        plt.plot(min_samples_leaf_numbers,min_samples_leaf_number_scores,'o-',color=colors[i],label=labels[i])
        i = i + 1
    plt.xlabel('Value of Minimal Samples Leaf Number for Decision Tree')
    plt.ylabel('Cross-Validated Accuracy')
    plt.legend()
    plt.show()

#for SVM
def compare_different_C():
    i = 0
    for dataset in datasets:
        C_numbers = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
        C_number_scores = []
        for C_number in C_numbers:
            classifier = SVC(kernel='rbf',C=C_number)
            scores = cross_val_score(classifier,dataset.data,dataset.target,cv=10,scoring='accuracy')
            C_number_scores.append(scores.mean())
        my_print(C_number_scores)
        x_axis = range(len(C_numbers))
        plt.plot(x_axis,C_number_scores,'o-',color=colors[i],label=labels[i])
        i = i + 1
    plt.xticks(x_axis, ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','2','3','4','5','6','7','8','9','10'], rotation=20)
    plt.xlabel('Value of C for SVM')
    plt.ylabel('Cross-Validated Accuracy')
    plt.legend()
    plt.show()

#for ANN
def compare_different_neuron_number():
    i = 0
    for dataset in datasets:
        first_hidden_layer_neuron_numbers = [1,2,3,4]
        first_hidden_layer_neuron_numbers.extend(range(5,101,5))
        neuron_number_scores = []
        for neuron_number in first_hidden_layer_neuron_numbers:
            classifier = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(neuron_number,), random_state=1, max_iter=200)
            scores = cross_val_score(classifier,dataset.data,dataset.target,cv=10,scoring='accuracy')
            neuron_number_scores.append(scores.mean())
        my_print(neuron_number_scores)
        plt.plot(first_hidden_layer_neuron_numbers,neuron_number_scores,'o-',color=colors[i],label=labels[i])
        i = i + 1
    plt.xlabel('Value of Neuron Number for Neural Network')
    plt.ylabel('Cross-Validated Accuracy')
    plt.legend()
    plt.show()
    
#for ANN again
def compare_different_hidden_layer_sizes_type():
    i = 0
    for dataset in datasets:
        hidden_layer_sizes_types = [(40,),(40,40),(40,40,40),(40,40,40,40),(40,40,40,40,40)]
        hidden_layer_sizes_types_scores = []
        for hidden_layer_sizes_type in hidden_layer_sizes_types:
            classifier = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=hidden_layer_sizes_type, random_state=1, max_iter=200)
            scores = cross_val_score(classifier,dataset.data,dataset.target,cv=10,scoring='accuracy')
            hidden_layer_sizes_types_scores.append(scores.mean())
        my_print(hidden_layer_sizes_types_scores)
        plt.plot([1,2,3,4,5],hidden_layer_sizes_types_scores,'o-',color=colors[i],label=labels[i])
        i = i + 1
    plt.xlabel('Value of Hidden Layer Number for Neural Network')
    plt.ylabel('Cross-Validated Accuracy')
    plt.legend()
    plt.show()

def my_print(arr):
    print '---begin---'
    my_str = "\n".join(str(var) for var in arr)
    print my_str
    print '---end---'


compare_different_hidden_layer_sizes_type()