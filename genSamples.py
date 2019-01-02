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
	f = open(dictPath, 'r')
	tokenList = f.read().split('\n')
	maxlength = int(tokenList[0])
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
			if i >= maxlength:
				break
			word = lines[i]
			if word in tokenList:
				tmp = tokenList.index(word)
				sample.append([tmp*1.0/len(tokenList)])
			else:
				print word				
		samples.append(sample)

	for i in range(len(samples)):
		if len(samples[i]) < maxlength:
			for j in range(len(samples[i]), maxlength):
				samples[i].append([0.0])
		if 'bug_' in filenames[i]:
			samples[i].append(1)
			samples[i].append(0)
		else:
			samples[i].append(0)
			samples[i].append(1)
	samples = np.array(samples)
	np.save(npyPath, samples)
	print 'done.'


if __name__ == '__main__':
	genSamples(sys.argv[1], sys.argv[2], sys.argv[3])
