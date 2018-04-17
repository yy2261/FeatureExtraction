import urllib
import time
import os
import sys

def download(downItem, storePath):
	githubPath = 'https://github.com'
	url = githubPath+downItem
	f = urllib.urlopen(url)
	print 'url read~'
	content = f.read()
	line = content.split('Branch:')[1].split('</span>')[0].split('-target">')[1]
	if line:
		path = url+'/archive/'+line+'.zip'
		try:
			urllib.urlretrieve(path, storePath+downItem.strip('/').replace('/', '.'))
			print downItem+' downloaded~'
		except:
			print 'something wrong with '+downItem+'!!!'
		time.sleep(5)
	else:
		print 'something wrong with '+str(url)+'!'


def main(namePath, storePath):
	f = open(namePath, 'r')
	for item in f.read().strip().split('\n'):
		download(item, storePath)
	f.close()
	print namePath+' done.'


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])