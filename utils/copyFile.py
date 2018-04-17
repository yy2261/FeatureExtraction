
import sys
import os


'''
write files in a directory into one file
'''

def main(path, writePath):
	names = os.listdir(path)
	with open(writePath, 'wb') as writeFd:
		for name in names:
			name = path+name
			with open(name, 'r') as readFd:
				for line in readFd:
					if line and line != ' \n':
						writeFd.write(line)



if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])