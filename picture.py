import matplotlib.pyplot as plt
'''
line20 = [39.9, 45.38, 48.88, 49.41, 43.49, 41.08, 35.69]
line50 = [39.14, 46.38, 49.55, 50.06, 46.60, 45.04, 41.68]
line100 = [39.53, 46.68, 49.28, 55.03, 51.66, 46.88, 40.97]
line200 = [40.01, 46.44, 51.09, 52.56, 49.41, 46.51, 38.83]
line500 = [39.13, 43.86, 45.71, 52.11, 48.57, 43.09, 38.98]

names = ['5', '10', '20', '50', '100', '200', '500']
x = range(len(names))

plt.figure()
plt.plot(x, line20, label='20')
plt.plot(x, line50, label='50')
plt.plot(x, line100, label='100')
plt.plot(x, line200, label='200')
plt.plot(x, line500, label='500')
plt.xlabel('number of hidden cells')
plt.ylabel('F1(%)')
plt.xticks(x, names)
plt.legend(title='vector length')
plt.show()
'''

name_list = ['camel-1.2\ncamel-1.4', 'camel-1.4\ncamel-1.6', 'jedit-3.2\njedit-4.0', 'jedit-4.0\njedit-4.1', 
		'log4j-1.0\nlog4j-1.1', 'lucene-2.0\nlucene-2.2', 'lucene-2.2\nlucene-2.4', 'xalan-2.4\nxalan-2.5', 
		'xerces-1.2\nxerces-1.3', 'synapse-1.1\nsynapse-1.2', 'poi-1.5\npoi-2.5', 'poi-2.5\npoi-3.0']
numlist1 = [32.36, 40.15, 47.32, 49.31, 52.51, 75.26, 75.11, 65.95, 26.10, 51.29, 80.91, 78.16]
numlist2 = [51.56, 46.29, 57.45, 62.42, 68.49, 68.38, 67.65, 59.38, 35.38, 51.49, 79.00, 78.76]

x = list(range(len(numlist1)))
total_width, n = 0.6, 2
width = total_width/n

plt.xticks([i+0.15 for i in x], name_list)

plt.bar(x, numlist1, width=width, label='tb-LSTM', ec='black', fc='w')

for i in range(len(x)):
	x[i] = x[i] + width

plt.bar(x, numlist2, width=width, label='Seml', ec='b',fc='w', hatch='/')

plt.ylabel('F1(%)')
plt.legend(prop={'size':12})
plt.show()

'''
name_list = ['jedit-4.1\ncamel-1.4', 'camel-1.4\njedit-4.1', 'log4j-1.1\njedit-4.1', 'jedit-4.1\nlog4j-1.1', 
		'lucene-2.2\nlog4j-1.1', 'lucene-2.2\nxalan-2.5', 'xerces-1.3\nxalan-2.5', 'xalan-2.5\nlucene-2.2', 
		'log4j-1.1\nlucene-2.2', 'xalan-2.5\nxerces-1.3', 'poi-3.0\nsynapse-1.2', 'synapse-1.2\npoi-3.0']
numlist1 = [32.06, 39.50, 39.17, 57.35, 57.85, 68.09, 67.76, 74.87, 74.87, 34.04, 50.24, 78.67]
numlist2 = [33.60, 45.75, 50.76, 63.29, 64.70, 64.33, 54.95, 66.06, 69.87, 35.00, 58.03, 78.64]

x = list(range(len(numlist1)))
total_width, n = 0.6, 2
width = total_width/n

plt.xticks([i+0.15 for i in x], name_list)

plt.bar(x, numlist1, width=width, label='tb-LSTM', ec='black', fc='w')

for i in range(len(x)):
	x[i] = x[i] + width

plt.bar(x, numlist2, width=width, label='Seml', ec='b', fc='w', hatch='/')
'''
plt.ylabel('F1(%)')
plt.legend(prop={'size':12})
plt.show()
