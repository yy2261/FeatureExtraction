import pandas
import os

def checkFileExists(featurePath, csvPath):
	f = open(csvPath, 'r')
	csv = pandas.read_csv(f)
	csvdirs = csv['dirname']
	defects = csv['bug']
	f.close()
	filenames = os.listdir(featurePath)
	for i in range(len(csvdirs)):
		exists = 0
		for filename in filenames:
			if csvdirs[i] in filename:
				if defects[i] > 0:
					os.system('mv '+featurePath+filename+' '+featurePath+'bug_'+filename)
				exists = 1
				break
		if exists == 0:
			print csvdirs[i]

if __name__ == '__main__':
	checkFileExists('/media/yy/10A4078410A40784/grad_proj/exp/camel_1.6.0/', '/media/yy/10A4078410A40784/grad_proj/exp/defectInfo/camel-1.6.csv')