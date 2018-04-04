import os
import sys
from tokenStem import *

'''
input: 	a folder with millions of feature files (x
		a txt file that store files already tokenized
output: one .txt file that features in each featureFile takes a line in the .txt
'''


def writeText(fd, lines):
	for i in range(len(lines)):
		if ' Declaration' in lines[i]:
			continue
		elif 'Statement' in lines[i] or 'Clause' in lines[i]:
			fd.write(' '+lines[i].split(' ')[0].lower())
		else:
			name = lines[i].split(': ')[-1]
			wordList = splittt(name)
			wordList = stemming(wordList)
			phrase = conj(wordList)
			if phrase == 'get_get':
				print name
			fd.write(' '+phrase)
	fd.write('\n')


def getFile(path):
	featureFiles = []
	dirs = os.listdir(path)
	for dir_ in dirs:
		featureFiles.append(path+dir_)
	print len(featureFiles)
	return featureFiles

def getText(path):
	featureFiles = getFile(path)
	g = open(sys.argv[2], 'wb')
	num = 0
	for featureFile in featureFiles:
		fd = open(featureFile, 'r')
		lines = fd.read().split('\n')
		fd.close()
		writeText(g, lines)
		sys.stdout.write(str(num)+'\r')
		sys.stdout.flush()
		num += 1 
	g.close()


if __name__ == '__main__':
	getText(sys.argv[1])
