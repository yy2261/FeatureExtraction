import urllib
import time
import os

def download(path, names):
	if len(names) == 0:
		return
	for name in names:
		newPath = path + name
		print newPath
		f = urllib.urlopen(newPath)
		lines = f.read().split('\n')
		nextNames = []
		for line in lines:
			if '-src' in line or 'source' in line:
				if '.tar' in line or '.zip' in line:
					try:
						package = line.split('">')[-1].split('</a>')[0]
						# check if package exists
						if os.path.exists('/home/yy/proj4wordvec/'+package) == True:
							print package+' already exists!'
						else:
							print 'downloading '+newPath+package+'...'
							urllib.urlretrieve(newPath+package, '/home/yy/proj4wordvec/'+package)
							time.sleep(5)
					except:
						print 'something wrong!'
						time.sleep(5)
					break
				elif '/</a>' in line and 'img' in line:
					nextName = line.split('/">')[1].split('</a>')[0]
					nextNames.append(nextName)
			elif '/</a>' in line and 'img' in line:
				nextName = line.split('/">')[1].split('</a>')[0]
				nextNames.append(nextName)
		download(newPath, nextNames)




path = 'https://mirror.tuna.tsinghua.edu.cn/apache/'
f = urllib.urlopen(path)
lines = f.read().split('Projects')[-1].split('\n')

names = []
for line in lines:
	if '/</a>' in line:
		name = line.split('/">')[1].split('</a>')[0]
		names.append(name)
download(path, names)