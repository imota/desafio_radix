import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

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
	return X, y

def print_results(y_test, test_predictions, clf):
	print(confusion_matrix(y_test,test_predictions))
	print(classification_report(y_test,test_predictions))
	print(clf.coef_)

def train():
	X, y = get_train_data()
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
	clf.fit(X_train, y_train)
	test_predictions = clf.predict(X_test)
	print_results(y_test, test_predictions, clf)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	#ax1.set_title("Mains power stability")    
	#ax1.set_xlabel('time')
	#ax1.set_ylabel('Mains voltage')
	ax1.plot(X,y, c='r', label='the data')
	leg = ax1.legend()
	plt.show()

	return clf

def test():
	data_test = pd.read_csv('p1_data_test.csv')
	X = [get_temps_of_line(i, data_test) for i in range(5208)]
	p1_predictions = clf.predict(X)
	
	df = pd.DataFrame(p1_predictions, columns = ['target'])
	df.to_csv('p1_predictions.csv', index = False)

if __name__ == "__main__":
	clf = LogisticRegression(class_weight = 'balanced', solver = 'lbfgs')
	train()
	test()