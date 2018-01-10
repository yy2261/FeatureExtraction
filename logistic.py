import numpy as np
import sys
from sklearn.linear_model import LogisticRegression


def loadData(trainSamplePath, trainLabelPath, testSamplePath, testLabelPath):
	trainX = np.load(trainSamplePath)
	trainY = np.load(trainLabelPath)
	trainY = np.argmin(trainY, axis=1)
	testX = np.load(testSamplePath)
	testY = np.load(testLabelPath)
	testY = np.argmin(testY, axis=1)
	return trainX, trainY, testX, testY

def lr(trainX, trainY, testX):
	cl = LogisticRegression()
	cl.fit(trainX, trainY)
	testPred = cl.predict(testX)
	return testPred

def calPRF(pred, y):
	zeros = np.zeros_like(y)
	ones = np.ones_like(y)

	tp = np.sum(np.logical_and(np.equal(pred, ones), np.equal(y, ones)).astype(int))
	fp = np.sum(np.logical_and(np.equal(pred, ones), np.equal(y, zeros)).astype(int))
	fn = np.sum(np.logical_and(np.equal(pred, zeros), np.equal(y, ones)).astype(int))

	if tp == 0:
		precision = 0
		recall = 0
		fmeasure = 0
	else:
		precision = tp*1.0/(tp+fp)
		recall = tp*1.0/(tp+fn)
		fmeasure = 2*precision*recall/(precision+recall)
	print 'precision: '+str(precision)
	print 'recall: '+str(recall)
	print 'f-measure: '+str(fmeasure)


if __name__ == '__main__':
	trainX, trainY, testX, testY = loadData(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	testPred = lr(trainX, trainY, testX)
	calPRF(testPred, testY)