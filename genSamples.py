import os
import csv
import sys
import numpy as np


def getVector(word, wordDict):
	key = word[0:2]
	if wordDict.has_key(key) == True:
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
		return ['0.000' for n in range(200)]
	else:
		print candidate.split(' ')[0]
		return candidate.split(' ')[1:-1]

def makeDict(f):
	items = f.read().split('\n')
	wordDict = {}
	for item in items:
		try:
			item.split(' ')[0].decode('ascii')
		except:
			continue
		key = item.split(' ')[0][0:2]
		if wordDict.has_key(key) == True:
			wordDict[key].append(item)
		else:
			wordDict[key] = [item]
	return wordDict

def genSamples(featurePath, dictPath, npyPath):
	f = open(dictPath)
	wordDict = makeDict(f)
	f.close()
	samples = []
	filenames = os.listdir(featurePath)
	for filename in filenames:
		print filename
		sample = []
		f = open(featurePath+filename, 'r')
		lines = f.read().split('\n')
		f.close()
		for i in range(len(lines)):
			if lines[i] == '':
				continue
			vector = getVector(lines[i], wordDict)
			for j in range(len(vector)):
				vector[j] = float(vector[j])
			sample.append(vector)
		if len(sample) > 200:
			sample = sample[:200]
		elif len(sample) < 200:
			for i in range(len(sample), 200):
				zeros = [0.000 for n in range(200)]
				sample.append(zeros)
		if 'bug_' in filename:
			sample.append(1)
			sample.append(0)
		else:
			sample.append(0)
			sample.append(1)
		samples.append(sample)
	samples = np.array(samples)
	np.save(npyPath, samples)
	print 'done.'


if __name__ == '__main__':
	genSamples(sys.argv[1], sys.argv[2], sys.argv[3])
