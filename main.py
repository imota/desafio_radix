import random
import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

def get_temps_of_line(i, data):
	X = np.ndarray(4)
	X[0] = data['Temp1'][i]
	X[1] = data['Temp2'][i]
	X[2] = data['Temp3'][i]
	X[3] = data['Temp4'][i]
	return X

def get_target(i, data):
	return data['target'][i]

def get_train_data():
	data_train = pd.read_csv('p1_data_train.csv')
	X = [get_temps_of_line(i, data_train) for i in range(5209)]
	y = [get_target(i, data_train) for i in range(5209)]

	plot(X,y)
	return X, y

def print_results(y_test, test_predictions, clf):
	print(confusion_matrix(y_test,test_predictions))
	print(classification_report(y_test,test_predictions))
	print(clf.coef_)

def plot(X,y):
	Xt = [X[i] for i in range(len(X)) if y[i] == True]
	Xf = [X[i] for i in range(len(X)) if y[i] == False]

	x0t, x0f = [random.choice(Xt)[0] for i in range(200)], [random.choice(Xf)[0] for i in range(200)]
	x1t, x1f = [random.choice(Xt)[1] for i in range(200)], [random.choice(Xf)[1] for i in range(200)]
	x2t, x2f = [random.choice(Xt)[2] for i in range(200)], [random.choice(Xf)[2] for i in range(200)]
	x3t, x3f = [random.choice(Xt)[3] for i in range(200)], [random.choice(Xf)[3] for i in range(200)]

	fig = plt.figure()
	ax1 = Axes3D(fig)
	ax1.plot(x0t, x1t, 'ro', zs = x2t, alpha = 0.5)
	ax1.plot(x0f, x1f, 'bo', zs = x2f, alpha = 0.5)
	plt.show()	

def train():
	X, y = get_train_data()
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
	clf.fit(X_train, y_train)
	test_predictions = clf.predict(X_test)
	print_results(y_test, test_predictions, clf)

	return clf

def test():
	data_test = pd.read_csv('p1_data_test.csv')

	X = [get_temps_of_line(i, data_test) for i in range(5208)]
	p1_predictions = clf.predict(X)
	
	df = pd.DataFrame(p1_predictions, columns = ['target'])
	df.to_csv('p1_predictions.csv', index = False, quoting = 1)

if __name__ == "__main__":
	clf = LogisticRegression(class_weight = 'balanced', solver = 'lbfgs')
	train()
	test()