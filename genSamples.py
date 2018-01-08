import os
import csv
import sys
import numpy as np

'''
generate sample files in the form of npy
input: feature path, vocabulary path, path to generated npy
'''

def writeTrain(samples, labels, npyPath):
	sample_npy = np.array(samples)
	label_npy = np.array(labels)
	np.save(npyPath+'Sample.npy', sample_npy)
	np.save(npyPath+'Label.npy', label_npy)
	print 'done.'

def genSamples(featurePath, dictPath, npyPath):
	'''
	generate samples from featurePath to .npy file
	length of the feature vector depends on the longest sequence in train file
	'''
	f = open(dictPath)
	lines = f.readlines()
	vecLength = int(lines[0])
	words = lines[1:]
	f.close()
	samples = []
	labels = []
	filenames = os.listdir(featurePath)
	for filename in filenames:
		sample = []
		f = open(featurePath+filename, 'r')
		lines = f.readlines()
		f.close()
		for i in range(len(lines)):
			word = lines[i]
			if word and word in words:
				sample.append(words.index(word)+1)
			else:
				sample.append(len(words)+1)
				words.append(word)
		if len(sample) > vecLength:
			sample = sample[:vecLength]
		elif len(sample) < vecLength:
			for i in range(len(sample), vecLength):
				sample.append(0)
		if 'bug_' in filename:
			labels.append([1, 0])
		else:
			labels.append([0, 1])
		samples.append(sample)
	writeTrain(samples, labels, npyPath)


if __name__ == '__main__':
	genSamples(sys.argv[1], sys.argv[2], sys.argv[3])
