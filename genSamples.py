import os
import csv
import sys
import numpy as np
from getVec import *

'''
generate sample files in the form of npy
input: feature path, vocabulary path, path to generated npy
'''

vecPath = '/media/yy/10A4078410A40784/grad_proj/exp/dicts/vec0625.txt'
featurePath = '/media/yy/10A4078410A40784/grad_proj/exp/stem_feature/'
npyPath = '/media/yy/10A4078410A40784/grad_proj/exp/vecDBNNpys/'


def genTrainSet(trainProj):
	wordDict = makeDict(vecPath)
	samples = []
	labels = []
	filenames = os.listdir(featurePath+trainProj)
	maxLength = 0
	for filename in filenames:
		print filename
		# add sample
		sample = []
		f = open(featurePath+trainProj+'/'+filename, 'r')
		lines = f.read().split('\n')
		f.close()
		if len(lines) > maxLength:
			maxLength = len(lines)
		for i in range(len(lines)):
			if lines[i] == '':
				continue
			vector = getVector(lines[i], wordDict)
			if vector == None:
				continue
			for j in range(len(vector)):
				sample.append(float(vector[j]))
		samples.append(sample)
		# add label
		if 'bug_' in filename:
			labels.append([1, 0])
		else:
			labels.append([0, 1])

	del wordDict

	for i in range(len(samples)):
		if len(samples[i]) < maxLength*100:
			for j in range(len(samples[i]), maxLength*100):
				samples[i].append(0.000)

	samples = np.array(samples)
	labels = np.array(labels)

	np.save(npyPath+trainProj+'.train.sample.npy', samples)
	np.save(npyPath+trainProj+'.train.label.npy', labels)

	return maxLength

def genTestSet(testProj, maxLength):
	wordDict = makeDict(vecPath)
	samples = []
	labels = []
	filenames = os.listdir(featurePath+testProj)
	for filename in filenames:
		print filename
		# add sample
		sample = []
		f = open(featurePath+testProj+'/'+filename, 'r')
		lines = f.read().split('\n')
		f.close()
		for i in range(len(lines)):
			if lines[i] == '':
				continue
			if len(sample) >= maxLength*100:
				break
			vector = getVector(lines[i], wordDict)
			if vector == None:
				continue
			for j in range(len(vector)):
				sample.append(float(vector[j]))
		samples.append(sample)
		# add label
		if 'bug_' in filename:
			labels.append([1, 0])
		else:
			labels.append([0, 1])

	del wordDict

	for i in range(len(samples)):
		if len(samples[i]) < maxLength*100:
			for j in range(len(samples[i]), maxLength*100):
				samples[i].append(0.000)

	samples = np.array(samples)
	labels = np.array(labels)

	np.save(npyPath+testProj+'.test.sample.npy', samples)
	np.save(npyPath+testProj+'.test.label.npy', labels)	


def genSamples(trainProj, testProj):
	'''
	generate samples from featurePath to .npy file
	length of the feature vector depends on the longest sequence in train file
	'''
	seqLen = genTrainSet(trainProj)
	genTestSet(testProj, seqLen)
	print 'done.'


if __name__ == '__main__':
	genSamples(sys.argv[1], sys.argv[2])
