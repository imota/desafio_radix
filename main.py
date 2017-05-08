import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def get_temps_of_line(i):
	X = np.ndarray(4)
	X[0] = data_train['Temp1'][i]
	X[1] = data_train['Temp2'][i]
	X[2] = data_train['Temp3'][i]
	X[3] = data_train['Temp4'][i]
	return X

def get_target(i):
	return data_train['target'][i]

def parse_train():
	X = []
	y = []
	for i in range(5209):
		X.append(get_temps_of_line(i))
		y.append(get_target(i))	
	return X, y

data_train = pd.read_csv('p1_data_train.csv')
X, y = parse_train()

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X,y)

results = clf.predict(X)
error = 0.
for i in range(5209):
	if results[i] != y[i]:
		error = error+1
print(error*100/5209)