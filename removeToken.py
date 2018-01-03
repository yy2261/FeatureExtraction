import os
import sys


def getFile(path, fileList):
	if os.path.isfile(path) == True:
		fileList.append(path)
		return
	else:
		path = path + '/'
		dirs = os.listdir(path)
		for dirr in dirs:
			newPath = path+dirr
			getFile(newPath, fileList)

def calToken(path, tokenDict):
	f = open(path, 'r')
	lines = f.read().split('\n')
	for line in lines:
		if line in tokenDict.keys():
			tokenDict[line] += 1
		else:
			tokenDict[line] = 1
	f.close()

def dict2List(tokenDict):
	tokenList = []
	for item in tokenDict.keys():
		if tokenDict[item] < 3:
			tokenList.append(item)
	return tokenList

def removeToken(file, writePath, tokenList):
	f = open(file, 'r')
	g = open(writePath+file.split('/')[-1], 'w')
	lines = f.read().split('\n')
	for line in lines:
		if line not in tokenList:
			g.write(line+'\n')
	f.close()
	g.close()



if __name__ == '__main__':
	tokenDict = {}
	writePath = sys.argv[2]
	fileList = []
	getFile(sys.argv[1], fileList)
	for file in fileList:
		calToken(file, tokenDict)
	tokenList = dict2List(tokenDict)
	print tokenList
	for file in fileList:
		removeToken(file, writePath, tokenList)