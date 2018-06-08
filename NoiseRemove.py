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

def CLNI(features, oldNoiseSet):
	NoiseSet = []
	for i in range(len(features)):
		disList = []
		for j in range(len(features)):
			if i == j or features[j] in oldNoiseSet:
				continue
			features[j].dis = calDistance(features[i], features[j])
			disList.append(features[j])
		disList.sort(key = lambda x:x.dis)
		num = 0
		for j in range(5):
			if features[i].label != disList[j].label:
				num += 1
		if num >= 3:
			print features[i].name
			NoiseSet.append(features[i])
	return NoiseSet

def calSimilarity(oldList, newList, num):
	if len(oldList) == 0 or len(newList) == 0:
		if num < 3:
			return 0
		else:
			return 1
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

	oldNoiseSet = []
	NoiseSet = []
	num = 1

	while calSimilarity(oldNoiseSet, NoiseSet, num) < 0.98:
		oldNoiseSet = NoiseSet[:]
		print '********************'
		print 'round '+str(num)
		num += 1
		NoiseSet = CLNI(features, oldNoiseSet)


	print "**************Noise List****************"
	for item in NoiseSet:
		print item.name
		os.system('mv '+path+item.name+' /media/yy/10A4078410A40784/grad_proj/exclude/')
	print 'list length = '+str(len(NoiseSet))
	os.system('supertux2')