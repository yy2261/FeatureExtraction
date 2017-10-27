
def labelNames(names):
	for name in names:
		if len(name.split(' ')) < 3:
			num = names.index(name)
			print name
			nameType = raw_input()
			if nameType == '-1':
				break
			else:
				names[num] = name.split(' ')[0]+' '+nameType


def rewrite(path, names):
	fd = open(path, 'w')
	for name in names:
		fd.write(name+'\n')
	fd.close()


def main():
	path = '/media/yy/10A4078410A40784/grad_proj/exp/featureName'
	fd = open(path, 'r')
	names = fd.read().split('\n')
	fd.close
	labelNames(names)
	rewrite(path, names)

if __name__ == '__main__':
	main()