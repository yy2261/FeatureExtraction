'''
yy grad proj source code
label files with bug info
'''


import os
import pandas
import sys


def selectFile(csvPath, featurePath):
	f = open(csvPath, 'r')
	csv = pandas.read_csv(f)
	csvdirs = csv['dirname']
	defects = csv['bug']
	f.close()
	defectNum = 0
	featuredirs = os.listdir(featurePath)
	for i in range(len(csvdirs)):
		fileName = csvdirs[i].split('.')[-1]
		dirName = csvdirs[i].split('.')[-2]
		secondDirName = csvdirs[i].split('.')[-3]
		num = 0
		fileList = []
		for featuredir in featuredirs:
			featurefileName = featuredir.split('.')[-2]
			featureDirName = featuredir.split('.')[-3]
			featureSecondDirName = featuredir.split('.')[-4]
			if fileName == featurefileName and dirName == featureDirName and secondDirName == featureSecondDirName:
				num += 1
				fileList.append(featuredir)
		if num == 1:
			if defects[i] == 1:
				os.system('mv '+featurePath+fileList[0]+' '+featurePath+'bug_'+fileList[0])
		elif num > 1:
			if defects[i] == 1:
				print num
				print csvdirs[i]
				print fileList
		elif num == 0 and defects[i] == 1:
			print csvdirs[i]
	print 'done.'


if __name__ == '__main__':
	selectFile(sys.argv[1], sys.argv[2])
