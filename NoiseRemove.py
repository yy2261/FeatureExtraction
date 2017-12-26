from __future__ import division
import os
import sys

class Distance(object):
	def __init__(self, name, distance):
		self.name = name
		self.dis = distance
		if 'bug_' in name:
			self.label = 1
		else:
			self.label = 0

def calDistance(oldFile, newFile):
	f = open(oldFile, 'r')
	oldLines = f.read().split('\n')
	f.close()
	f= open(newFile, 'r')
	newLines = f.read().split('\n')
	f.close()
	matrix = [[i+j for j in range(len(newLines) + 1)] for i in range(len(oldLines) + 1)]
	for i in xrange(1,len(oldLines)+1):  
		for j in xrange(1,len(newLines)+1):  
			if oldLines[i-1] == newLines[j-1]:  
				d = 0  
			else:  
				d = 1  
			matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)  
	return matrix[len(oldLines)][len(newLines)]  

def CLNI(path):
	files = os.listdir(path)
	NoiseSet = []
	for i in range(len(files)):
		tmp = files[i]
		files[i] = files[0]
		files[0] = tmp
		disI = Distance(files[0], 0)
		disList = []
		for j in range(1, len(files)):
			if files[j] in NoiseSet:
				continue
			dis = calDistance(path+files[0], path+files[j])
			disList.append(Distance(files[j], dis))
		disList.sort(key = lambda x:x.dis)
		num = 0
		for j in range(5):
			if disI.label != disList[j].label:
				num += 1
		if num >= 3:
			NoiseSet.append(files[i])
	return NoiseSet

def calSimilarity(oldList, newList):
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
	oldNoiseSet = CLNI(path)
	NoiseSet = CLNI(path)
	for j in range(100):
		if calSimilarity(oldNoiseSet, NoiseSet) > 0.99:
			break
		else:
			oldNoiseSet = NoiseSet[:]
			NoiseSet = CLNI(path)
	for item in NoiseSet:
		os.system('mv '+path+item+' exclude/')
