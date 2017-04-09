import numpy as np
import matplotlib.pyplot as plt
import json


result_file = open("result.json","r")
result = json.load(result_file)
result_file.close()

common_x_axis_names = result.keys()
#common_x_axis = range(len(common_x_axis_names))


decision_tree_scores = []
svm_scores = []
neural_network_scores = []


for dataset_name in result.keys():
    decision_tree_scores.append(result[dataset_name]['decision_tree'])
    svm_scores.append(result[dataset_name]['svm'])
    neural_network_scores.append(result[dataset_name]['neural_network'])

bar_width = 0.2
index = np.arange(len(common_x_axis_names))

plt.bar(index,decision_tree_scores,bar_width,alpha=0.75,color='r',label='Decision Tree')
plt.bar(index+bar_width,svm_scores,bar_width,alpha=0.75,color='g',label='SVM')
plt.bar(index+2*bar_width,neural_network_scores,bar_width,alpha=0.75,color='b',label='Neural Network')
#plt.xticks(common_x_axis, common_x_axis_names, rotation=20)

plt.xticks(index + bar_width, common_x_axis_names)
plt.legend(loc="Algorithms Comparison")
plt.xlabel("Dataset Name")
plt.ylabel("Cross Validation F1_Macro")

plt.show()