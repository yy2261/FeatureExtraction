 #-*-coding: utf-8 -*-
import six.moves.cPickle as pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys

def train_logistic(dataset=None):
	if dataset==None:
		print('Dataset empty!')
		return
	data = np.load(dataset)
	trainX = []
	trainY = []
	for item in data:
		sample = []
		for i in range(200):
			for number in item[i]:
				sample.append(number)
		if item[200] == 1:
			trainX.append(sample)
			trainY.append(item[200])
			trainX.append(sample)
			trainY.append(item[200])
			trainX.append(sample)
			trainY.append(item[200])
		trainX.append(sample)
		trainY.append(item[200])
	print trainY

	clf=LogisticRegression(penalty='l2', C=500, solver='liblinear')
	clf.fit(trainX, trainY)
	with open('models/LR.pkl','wb') as f1:
		pickle.dump(clf,f1)

def measure(result, Y):
	tp, fp, tn, fn = 0, 0, 0, 0
	for i in range(len(result)):
		if result[i] == 1 and Y[i] == 1:
			tp += 1
		elif result[i] == 1 and Y[i] == 0:
			fp += 1
		elif result[i] == 0 and Y[i] == 1:
			fn += 1
		elif result[i] == 0 and Y[i] == 0:
			tn += 1
	precision = tp*1.0/(tp+fp)
	recall = tp*1.0/(tp+fn)
	f1 = 2*precision*recall/(precision+recall)
	print 'precision:'+str(precision)
	print 'recall:'+str(recall)
	print 'f1:'+str(f1)

def predict(model_pkl, dataset):
	with open(model_pkl,'rb') as f1:
		clf=pickle.load(f1)
	if dataset==None:
		print('Dataset empty!')
		return
	data = np.load(dataset)
	testX = []
	testY = []
	for item in data:
		sample = []
		for i in range(200):
			for number in item[i]:
				sample.append(number)
		testX.append(sample)
		testY.append(item[200])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
	result=clf.predict(testX)
	measure(result, testY)

def main():
	train_logistic(sys.argv[1])
	predict('models/LR.pkl', sys.argv[2])


if __name__ == '__main__':
	main()