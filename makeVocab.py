import os
import sys

'''
generate vocabulary for with-project test features
length of the longest feature sequence in line 1

input: feature path, path to vocabulary
'''

def getFile(path, files):
	items = os.listdir(path)
	for item in items:
		newPath = path+item
		if os.path.isfile(newPath):
			if '.java' in newPath:
				files.append(newPath)
		else:
			getFile(newPath+'/', files)


def makeVocab(path, dictPath):
	files = []
	getFile(path, files)
	words = []
	maxNum = 0
	for file in files:
		f = open(file, 'r')
		lines = f.read().split('\n')
		if len(lines) > maxNum:
			maxNum = len(lines)
		for i in range(len(lines)):
			if lines[i] and lines[i] not in words:
				words.append(lines[i])
		f.close()
	f = open(dictPath, 'w')
	f.write(str(maxNum)+'\n')
	for word in words:
		f.write(word+'\n')
	f.close()
	print 'done.'


if __name__ == '__main__':
	makeVocab(sys.argv[1], sys.argv[2])
