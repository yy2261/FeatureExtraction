import os
import sys

'''
split Type Declaration to a new file
args: projectDir
read one file, get path, get each type name, split into lists, and write into different files
'''

def main(featureDir):
	files = os.listdir(featureDir)
	for item in files:
		f = open(featureDir+item, 'r')
		content = f.read()
		f.close()
		blocks = content.split('Type Declaration')
		if len(blocks) == 2:
			continue
		pathName = '.'.join(item.split('.')[:-2])
		for i in range(len(blocks)):
			if blocks[i]:
				className = blocks[i].split('\n')[0].split(': ')[1]
				f = open(featureDir+pathName+'.'+className+'.java', 'wb')
				f.write('Type Declaration'+blocks[i])
				f.close()



if __name__ == '__main__':
	main(sys.argv[1])