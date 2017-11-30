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
		num = 0
		fileList = []
		for featuredir in featuredirs:
			featurefileName = featuredir.split('.')[-2]
			if fileName == featurefileName:
				num += 1
				fileList.append(featuredir)
		if num == 1:
			if defects[i] == 1:
				os.system('mv '+featurePath+fileList[0]+' '+featurePath+'bug_'+fileList[0])
		if num > 1:
			if defects[i] == 1:
				print num
				print csvdirs[i]
				print fileList


if __name__ == '__main__':
	selectFile(sys.argv[1], sys.argv[2])
