import re
import sys
from nltk.stem.lancaster import LancasterStemmer

def parseCamel(word, tokens):
	isComplete = 1
	for i in range(len(word)-1):
		if word[i].islower() == 1 and word[i+1].isupper() == 1:
			isComplete = 0
			wordNew = word[0:i+1]
			wordToBeSplit = word[i+1:]
			tokens.append(wordNew)
			parseCamel(wordToBeSplit, tokens)
			break
	if isComplete == 1:
		tokens.append(word)
		return

def processNums(word):
	wordList = []
	result = word
	while result:
		match = re.search("[0-9]*([a-z]*)[0-9]*([0-9+a-z]*)", result)
		result = match.group(2)
		if match.group(1):
			wordList.append(match.group(1))
	return wordList			

def splittt(phrase):
	if '.' in phrase:
		phrase = phrase.split('.')[-1].strip(' ')
	if 'statement' in phrase:
		phrase = phrase.split('statement')[0].strip(' ')
	if 'clause' in phrase:
		phrase = phrase.split('clause')[0].strip(' ')
	if '_' in phrase:
		words = phrase.split('_')
	else:
		words = [phrase]
	
	newWords = []
	for word in words:
		if word.islower() == 1 or word.isupper() == 1:
			newWords.append(word)
		else:
			tokens = []
			parseCamel(word, tokens)
			for token in tokens:
				newWords.append(token)
	for i in range(len(newWords)):
		newWords[i] = newWords[i].lower()
		result = processNums(newWords[i])
		if len(result) == 0:
			continue
		elif len(result) == 1:
			newWords[i] = result[0]
		else:
			newWords[i] = result[0]
			for j in range(1, len(result)):
				newWords.append(result[j])
	return newWords

def stemming(wordList):
	newList = []
	st = LancasterStemmer()
	for word in wordList:
		newList.append(st.stem(word))
	return newList

def conj(wordList):
	phrase = ''
	wordList.sort()
	for i in range(len(wordList)):
		phrase += wordList[i]+'_'
	return phrase.strip('_')


if __name__ == '__main__':
	wordList = splittt(sys.argv[1])
	wordList = stemming(wordList)
	phrase = conj(wordList)
	print phrase