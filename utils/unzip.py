import os
import sys


def main(path):
	packages = os.listdir(path)
	f = open('/home/yy/log.txt', 'w')
	for package in packages:
		name = path+package
		try:
			if 'tar.gz' in name:
				os.system('tar -zxvf '+name+' -C /media/yy/10A4078410A40784/grad_proj/data/0')
			elif 'tar.bz' in name:
				os.system('tar -xvf '+name+' -C /media/yy/10A4078410A40784/grad_proj/data/0')
			elif '.zip' in name:
				os.system('unzip '+name+' -d /media/yy/10A4078410A40784/grad_proj/data/0')
			else:
				f.write(name+' failed!\n')
		except:
			f.write(name+' failed!\n')
	f.close()

if __name__ == '__main__':
	main(sys.argv[1])