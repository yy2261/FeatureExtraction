import os
import sys

def removeType(word):
	newWord = ''
	if 'Statement' in word:
			newWord = word.split(' Statement')[0]
	elif 'Clause' in word:
		newWord = word.split(' Clause')[0]
	if ': ' in word:
		newWord = word.split(': ')[1]
	if '<' in newWord:
		newWord = newWord.split('<')[0]
	return newWord

def main(path, newPath):
	files = os.listdir(path)
	for file in files:
		f = open(path+file, 'r')
		g = open(newPath+file, 'w')
		lines = f.read().split('\n')
		for line in lines:
			newLine = removeType(line)
			g.write(newLine+'\n')
		f.close()
		g.close()

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])