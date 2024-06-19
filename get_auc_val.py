from audioop import mul
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from itertools import cycle
import matplotlib.pyplot as plt

file_path = "D:\MTP\AUC\model_found_results\samp15temp5\\auc_data.txt"
def softmax(np_array):
 
	for i in range(np_array.shape[0]):
		exp_values = np.exp(np_array[i])
	
		# Computing sum of these values
		exp_values_sum = np.sum(exp_values)
	
		# Returing the softmax output.
		np_array[i] = exp_values/exp_values_sum
	return np_array

fptr = open(file_path, 'r')
lis = fptr.readlines()
fptr.close()
val = []
get_orig = []
for i in range(len(lis)):
	temp = lis[i].rstrip().split(" ")
	val.append([float(temp[0]),float(temp[1])])
	get_orig.append(int(temp[2]))

y_pred = np.array(val)
y_pred_prob = softmax(y_pred)
print(y_pred_prob.shape)

y_true = []
for i in range(len(get_orig)):
	p = get_orig[i]
	temp = [0,0]
	if p == 1:
		temp[p] = 1
		y_true.append(temp)
	else:
		temp[p] = 1
		y_true.append(temp)

#print(y_true)
y_true = np.array(y_true)
print(y_true.shape)

auc = roc_auc_score(y_true, y_pred_prob, multi_class="ovr")
print("Auc score:",auc)

fig, ax = plt.subplots(figsize=(5, 5))
target_names = ['Fake','Real']
# print(y_true[:, 0])
# print(y_pred_prob[:, 0])
RocCurveDisplay.from_predictions(
	y_true[:, 0],
	y_pred_prob[:, 0],
	name=f"ROC curve for {target_names[0]}",
	color="orange",
	ax=ax,
	
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
#plt.show()

# print(y_true[:, 1])
# print(y_pred_prob[:, 1])
#fig, ax = plt.subplots(figsize=(5, 5))
RocCurveDisplay.from_predictions(
	y_true[:, 1],
	y_pred_prob[:, 1],
	name=f"ROC curve for {target_names[1]}",
	color="blue",
	ax=ax,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\n Real and Fake")
plt.legend()
plt.show()