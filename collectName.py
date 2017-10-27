import os


def collectFile(path):
	files = []
	feaPaths = os.listdir(path)
	for feaPath in feaPaths:
		if '_' in feaPath:
			filenames = os.listdir(path+feaPath+'/')
			for filename in filenames:
				filepath = path+feaPath+'/'+filename
				files.append(filepath)
	return files


def collectName(file, names):
	fd = open(file, 'r')
	lines = fd.read().split('\n')
	for line in lines:
		if 'Method Invocation' in line:
			name = line.split(': ')[1]
			if name not in names:
				names.append(name)
	fd.close()


def writeName(names, path):
	fd = open(path, 'w')
	for name in names:
		fd.write(name+'\n')
	fd.close()

def main():
	path = '/media/yy/10A4078410A40784/grad_proj/exp/'
	writePath = '/media/yy/10A4078410A40784/grad_proj/exp/featureName'
	files = collectFile(path)
	names = []
	for file in files:
		collectName(file, names)
	writeName(names, writePath)

if __name__ == '__main__':
	main()