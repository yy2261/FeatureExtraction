from __future__ import division
import threading
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

def CLNI(features, fileBegin, fileNum):
	global NoiseSet
	global oldNoiseSet
	for i in range(fileBegin, fileBegin+fileNum):
		disList = []
		for j in range(len(features)):
			if features[j] in oldNoiseSet:
				continue
			features[j].dis = calDistance(features[i], features[j])
			disList.append(features[j])
		disList.sort(key = lambda x:x.dis)
		num = 0
		num_label = 0
		while num_label < 5 and num < len(disList):
			if features[i].label != disList[num].label:
				num_label += 1
			num += 1 
		if num_label / num >= 0.6:
			print features[i].name
			NoiseSet.append(features[i])

def calSimilarity(oldList, newList):
	if len(oldList) == 0 or len(newList) == 0:
		return 0
	length = max(len(oldList), len(newList))
	num = 0
	for i in range(len(oldList)):
		for j in range(len(newList)):
			if oldList[i] == newList[j]:
				num += 1
	rate = num / length
	return rate


if __name__ == '__main__':
	path = sys.argv[1]
	files = os.listdir(path)
	features = []
	for file in files:
		f = open(path+file, 'r')
		featureList = f.read().split('\n')
		feature = Feature(file, 0, featureList)
		features.append(feature)

	global oldNoiseSet
	global NoiseSet
	oldNoiseSet = []
	NoiseSet = []

	num = 1
	lendir = int(sys.argv[2])

	while calSimilarity(oldNoiseSet, NoiseSet) < 0.95:
		oldNoiseSet = NoiseSet[:]
		NoiseSet = []

		print 'once again.................................................'
		print 'round '+str(num)
		num += 1
		t1 = threading.Thread(target=CLNI, args=(features, 0, lendir//4))
		t2 = threading.Thread(target=CLNI, args=(features, lendir//4, lendir//4))
		t3 = threading.Thread(target=CLNI, args=(features, lendir//2, lendir//4))
		t4 = threading.Thread(target=CLNI, args=(features, lendir//4 *3, lendir-lendir//4*3))
		threads = [t1, t2, t3, t4]

		for t in threads:
			t.setDaemon(True)
			t.start()
		t.join()


	print "**************Noise List****************"
	for item in NoiseSet:
		print item.name
#		os.system('mv '+path+item.name+' exclude/')