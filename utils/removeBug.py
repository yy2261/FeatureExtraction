import os
import sys

path = sys.argv[1]
fileNames = os.listdir(path)
for fileName in fileNames:
	if 'bug_' in fileName:
		newName = fileName.strip('bug_')
		os.system('mv '+path+fileName+' '+path+newName)