import os
import csv
import sys
import numpy as np


def getVector(word, wordDict):
	key = word[0:2]
	if key in wordDict:
		for text in wordDict[key]:
			vectorName = text.split(' ')[0]
			if word == vectorName:
				vector = text.split(' ')[1:-1]
				return vector
	candidate = ''
	maxcount = 0
	for key in wordDict.keys():
		if key not in word:
			continue
		for item in wordDict[key]:
			isContain = 1
			dict_words = item.split(' ')[0].split('_')
			for dict_word in dict_words:
				if dict_word not in word.split('_'):
					isContain = 0
					break
			if isContain == 1:
				if len(dict_words) > maxcount and (len(dict_words) == 1 or dict_words[0] != dict_words[1]):
					candidate = item
					maxcount = len(dict_words)
	if candidate == '':
		print 'mismatch!'
		return None
	else:
		print candidate.split(' ')[0]
		return candidate.split(' ')[1:-1]

def makeDict(dictPath):
	wordDict = {}
	with open(dictPath, 'r') as f:
		for line in f:
			try:
				line.strip().split(' ')[0].decode('ascii')
			except:
				continue
			key = line.split(' ')[0][0:2]
			if key in wordDict:
				wordDict[key].append(line)
			else:
				wordDict[key] = [line]
	print len(wordDict)
	return wordDict

def genSamples(featurePath, dictPath, npyPath):
	wordDict = makeDict(dictPath)
	samples = []
	filenames = os.listdir(featurePath)
	maxlength = 0
	for filename in filenames:
		print filename
		sample = []
		f = open(featurePath+filename, 'r')
		lines = f.read().split('\n')
		f.close()
		if len(lines) > maxlength:
			maxlength = len(lines)
		for i in range(len(lines)):
			if lines[i] == '':
				continue
			vector = getVector(lines[i], wordDict)
			if vector == None:
				continue
			for j in range(len(vector)):
				vector[j] = float(vector[j])
			sample.append(vector)
		samples.append(sample)

	newSamples = []
	for i in range(len(samples)):
		if len(samples[i]) < maxlength:
			for j in range(len(sample), maxlength):
				zeros = [0.000 for n in range(200)]
				sample.append(zeros)
		if 'bug_' in filenames[i]:
			sample.append(1)
			sample.append(0)
		else:
			sample.append(0)
			sample.append(1)
		newSamples.append(sample)
	samples = np.array(newSamples)
	np.save(npyPath, samples)
	print 'done.'


if __name__ == '__main__':
	genSamples(sys.argv[1], sys.argv[2], sys.argv[3])
