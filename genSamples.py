import os
import csv

def processWord(word):
	newWord = ''
	if 'Statement' in word:
			newWord = word.split(' Statement')[0]
	elif 'Clause' in word:
		newWord = word.split(' Clause')[0]
	if ':' in word:
		newWord = word.split(': ')[1].split('<')[0]
	return newWord

def writeTrain(rows, csvPath):
	with open(csvPath, 'w') as csvfile:
		writer = csv.writer(csvfile)
		for row in rows:
			if 'bug_' in row[-1]:
				row.append(1)
				row.append(0)
			else:
				row.append(0)
				row.append(1)
			writer.writerow(row)
	print 'done.'

def genSamples(featurePath, dictPath, csvPath):
	f = open(dictPath)
	words = f.readlines()
	f.close()
	samples = []
	filenames = os.listdir(featurePath)
	for filename in filenames:
		sample = []
		f = open(featurePath+filename, 'r')
		lines = f.readlines()
		f.close()
		for i in range(len(lines)):
			word = processWord(lines[i])
			if word and word in words:
				sample.append(words.index(word)+1)
			else:
				sample.append(len(words)+1)
				words.append(word)
		if len(sample) > 200:
			sample = sample[:200]
		elif len(sample) < 200:
			for i in range(len(sample), 200):
				sample.append(0)
		sample.append(filename)
		samples.append(sample)
	writeTrain(samples, csvPath)


if __name__ == '__main__':
	genSamples('/media/yy/10A4078410A40784/grad_proj/exp/camel_1.6.0/', '/media/yy/10A4078410A40784/grad_proj/exp/dicts/dict_camel_1.2.0', '/media/yy/10A4078410A40784/grad_proj/exp/dicts/camel_test_1.6.0.csv')
