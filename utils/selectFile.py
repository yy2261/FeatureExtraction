'''
yy grad proj utils
select files included in csv from feature files.
input
	csv path
	feature path
	target new dir path

strategy:

check filename, for the classes share the same filename, append to a list

check the list and check the dirName and secondDirName, if both are the same then print the path

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
		try:
			secondDirName = csvdir.split('.')[-3]
		except:
			pass
		num = 0
		fileList = []
		for featuredir in featuredirs:
			featurefileName = featuredir.split('.')[-2]
			if fileName == featurefileName:
				num += 1
				fileList.append(featuredir)
		if num == 1:
			os.system('cp '+featurePath+fileList[0]+' '+targetPath)
		elif num > 1:
			tmpFileList = []
			for item in fileList:
				featureDirName = item.split('.')[-3]
				if dirName != featureDirName:
					num -= 1
				else:
					tmpFileList.append(item)
			fileList = tmpFileList
		if num == 1:
			os.system('cp '+featurePath+fileList[0]+' '+targetPath)
		elif num > 1:
			tmpFileList = []
			for item in fileList:
				featureSecondDirName = item.split('.')[-4]
				if secondDirName != featureSecondDirName:
					num -= 1
				else:
					tmpFileList.append(item)
			fileList = tmpFileList
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