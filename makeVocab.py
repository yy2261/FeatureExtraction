import os

def makeVocab(path, dictPath):
	files = os.listdir(path)
	words = []
	for file in files:
		f = open(path+file, 'r')
		lines = f.read().split('\n')
		word = ''
		for i in range(len(lines)):
			if 'Statement' in lines[i]:
				word = lines[i].split(' Statement')[0]
			elif 'Clause' in lines[i]:
				word = lines[i].split(' Clause')[0]
			if ':' in lines[i]:
				word = lines[i].split(': ')[1].split('<')[0]
			if word and word not in words:
				words.append(word)
		f.close()
	f = open(dictPath, 'w')
	for word in words:
		f.write(word+'\n')
	f.close()
	print 'done.'


if __name__ == '__main__':
	makeVocab('/media/yy/10A4078410A40784/grad_proj/exp/camel_1.6.0/', '/media/yy/10A4078410A40784/grad_proj/exp/dicts/dict_camel_1.6.0')