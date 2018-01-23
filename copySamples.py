import os
import sys

class Feature(object):
	def __init__(self, name, distance, featureList):
		self.name = name
		self.dis = distance
		self.list = featureList
		if 'bug_' in name:
			self.label = 1
		else:
			self.label = 0

def calDistance(oldFeatures, newFeatures):
	oldLine = len(oldFeatures.list)
	newLine = len(newFeatures.list)
	matrix = [[i+j for j in range(newLine + 1)] for i in range(oldLine + 1)]
	for i in xrange(1, oldLine + 1):  
		for j in xrange(1, newLine + 1):  
			if oldFeatures.list[i-1] == newFeatures.list[j-1]:  
				d = 0  
			else:  
				d = 1  
			matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)  
	return matrix[oldLine][newLine]

def fileCopy(path, features, k):
	num = 0
	for item in features:
		if item.label == 1:
			disList = []
			for i in range(len(features)):
				features[i].dis = calDistance(features[i], item)
				disList.append(features[i])
			disList.sort(key = lambda x:x.dis)
			for i in range(k):
				os.system('cp '+path+disList[k].name+' '+path+str(num)+disList[k].name)
				num += 1

def main(path):
	Features = []
	files = os.listdir(path)
	for file in files:
		f = open(path+file, 'r')
		lines = f.read().split('\n')
		f.close()
		feature = Feature(file, 0, lines)
		Features.append(feature)
	fileCopy(path, Features, 5)


if __name__ == '__main__':
	main(sys.argv[1])