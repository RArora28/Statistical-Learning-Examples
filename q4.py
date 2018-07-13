#!/usr/bin/env python

import sys
import os
import numpy as np
from collections import *
from operator import itemgetter

classes = ['galsworthy/','galsworthy_2/','mill/','shelley/','thackerey/','thackerey_2/','wordsmith_prose/','cia/','johnfranklinjameson/','diplomaticcorr/']

# self.X: list of feature vectors(dicts).
# self.Y: list of classifiers.
class FeatureVector(object):
	def __init__(self,vocabsize,numdata):
		self.vocabsize = vocabsize
		self.X, self.Y = [], []

	def make_featurevector(self, input, classid):
		self.Y.append(classid)
		self.X.append(defaultdict(int))
		# frequency calculation.
		latest = len(self.X) - 1
		for word in input: 
			self.X[latest][word] += 1
		# normalisation of the word count
		l = len(input)
		for word in self.X[latest]: 
			self.X[latest][word] = float(float(self.X[latest][word]) / float(l))

class KNN(object):
	def __init__(self,trainVec,testVec):
		self.X_train = trainVec.X
		self.Y_train = trainVec.Y
		self.X_test  = testVec.X
		self.Y_test  = testVec.Y
		self.metric  = Metrics('accuracy')

	def classify(self, nn = 1):
		testlen  = len(self.X_test)
		trainlen = len(self.X_train)
		correct  = 0
		for j in range(0, testlen):
			manhattan = defaultdict(list)
			for i in range(0, trainlen): 
				manhattan[i] = [0, self.Y_train[i]]
				for word in self.X_train[i]: 
					if word in self.X_test[j]: manhattan[i][0] += abs(self.X_train[i][word] - self.X_test[j][word])
					else: manhattan[i][0] += self.X_train[i][word]
				for word in self.X_test[j]:
					if word in self.X_train[i]: pass # already considered above	
					else: manhattan[i][0] += self.X_test[j][word]

			# found distances of test j from all training samples:
			k_nearest = sorted(manhattan.items(), key = itemgetter(1))
			k_nearest = k_nearest[:nn]
			cnt = defaultdict(int)

			# print j, k_nearest
			for n in k_nearest: 
				cnt[n[1][1]] += 1

			max_count = 0
			predicted_class = 0

			for c in cnt: 
				if cnt[c] > max_count: 
					max_count = cnt[c]
					predicted_class = c


			print classes[predicted_class-1].strip('/')
			# correct += (str(predicted_class) == str(self.Y_test[j]))			

		# print "Accuracy :", float(correct)*100/float(j+1), "%  with K: ", nn

class Metrics(object):
	def __init__(self,metric):
		self.metric = metric

	def score(self):
		if self.metric == 'accuracy':
			return self.accuracy()
		elif self.metric == 'f1':
			return self.f1_score()

	def get_confmatrix(self,y_pred,y_test):
		"""
		Implements a confusion matrix
		"""

	def accuracy(self):
		"""
		Implements the accuracy function
		"""

	def f1_score(self):
		"""
		Implements the f1-score function
		"""

if __name__ == '__main__':
	traindir = sys.argv[1]
	testdir = sys.argv[2]
	inputdir = [traindir, testdir]

	vocab = 30000 # Random Value
	trainsz = 1000 # Random Value
	testsz = 500 # Random Value

	# print('Making the feature vectors.')
	trainVec = FeatureVector(vocab,trainsz)
	testVec = FeatureVector(vocab,testsz)

	for idir in inputdir:
		classid = 1
		for c in classes:
			inputs = []
			listing = os.listdir(idir+c)
			for filename in listing:
				f = open(idir + c + filename, 'r')
				lines = f.readlines()
				inputs = []
				for line in lines: 
					line = line.split(' ')
					for word in line:
						inputs.append(word)

				if idir == traindir:
					trainVec.make_featurevector(inputs,classid)
				else:
					testVec.make_featurevector(inputs,classid)
			classid += 1
	
	# print(trainVec.X.shape,trainVec.Y.shape,testVec.X.shape,testVec.Y.shape)

	knn = KNN(trainVec,testVec)
	knn.classify(7)
