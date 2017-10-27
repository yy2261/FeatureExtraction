import urllib


def getPackages(path):
	packages = []
	fd = urllib.urlopen(path)
	lines = fd.readlines()
	for line in lines:
		if '<td class="colFirst">' in line:
			package = line.split('</a>')[0].split('>')[-1]
			packages.append(package)
	fd.close()
	return packages

def getClass(path):
	classes = []
	fd = urllib.urlopen(path)
	lines = fd.readlines()
	for line in lines:
		if '<td class="colFirst">' in line:
			classs = line.split('</a>')[0].split('>')[-1]
			classes.append(classs)
	fd.close()
	return classes

def getMethod(path):
	methods = []
	fd = urllib.urlopen(path)
	lines = fd.readlines()
	for line in lines:
		if 'meta name="keywords"' in line and '(' in line:
			method = line.split('="')[-1].split('()">')[0]
			methods.append(method)
	fd.close()
	return methods

def main():
	path = 'https://docs.oracle.com/javase/8/docs/api/'
	packages = getPackages(path+'overview-summary.html')
	writePath = '/media/yy/10A4078410A40784/grad_proj/exp/APINames_3'
	fd = open(writePath, 'w')
	index = packages.index('javax.swing.filechooser')
	for i in range(index, len(packages)):
		print 'package: '+packages[i]
		classes = getClass(path+packages[i].replace('.', '/')+'/package-summary.html')
		for classs in classes:
			print 'class: '+classs
			methods = getMethod(path+packages[i].replace('.', '/')+'/'+classs+'.html')
			if methods:
				for method in methods:
					print method
					fd.write(method+'\n')
	fd.close()


if __name__ == '__main__':
	main()