import os
import csv
import sys
from tokenStem import *

def processWord(word):
	wordList = splittt(word)
	wordList = stemming(wordList)
	phrase = conj(wordList)
	return phrase

def writeTrain(rows, csvPath):
	with open(csvPath, 'w') as csvfile:
		writer = csv.writer(csvfile)
		for row in rows:
			if 'bug_' in row[-1]:
				row.append(1)
				row.append(0)
			else:
				row.append(0)
				row.append(1)
			writer.writerow(row)
	print 'done.'

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

def genSamples(featurePath, stemPath):
	samples = []
	filenames = os.listdir(featurePath)
	for filename in filenames:
		print filename
		g = open(stemPath+filename, 'wb')
		f = open(featurePath+filename, 'r')
		lines = f.read().split('\n')
		f.close()
		for i in range(len(lines)):
			if lines[i] == '':
				continue
			word = processWord(lines[i])
			if word == '':
				continue
			g.write(word+'\n')
		g.close()
	print 'done.'


if __name__ == '__main__':
	genSamples(sys.argv[1], sys.argv[2])
