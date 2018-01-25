'''
yy grad proj utils
select files included in csv from feature files.
input
	csv path
	feature path
	target new dir path
'''

import os
import pandas
import sys


def selectFile(csvPath, featurePath, targetPath):
	f = open(csvPath, 'r')
	csv = pandas.read_csv(f)
	csvdirs = csv['dirname']
	defects = csv['bug']
	f.close()
	defectNum = 0
	for defect in defects:
		if defect == 1:
			defectNum += 1
	print defectNum
	featuredirs = os.listdir(featurePath)
	for csvdir in csvdirs:
		fileName = csvdir.split('.')[-1]
		dirName = csvdir.split('.')[-2]
		secondDirName = csvdir.split('.')[-3]
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
			os.system('cp '+featurePath+fileList[0]+' '+targetPath)
		elif num > 1:
			print num
			print csvdir
			print fileList
		elif num == 0:
			print csvdir
	print 'done.'


if __name__ == '__main__':
	selectFile(sys.argv[1], sys.argv[2], sys.argv[3])