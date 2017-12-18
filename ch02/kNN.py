#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import operator

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	# Calculation distance
	dataSetSize = dataSet.shape[0]
	# After tile()
	# inX = ([1, 1],
	#		 [1, 1],
	#		 [1, 1],
	#		 [1, 1])
# 	>>> b=[1,3,5]
# 	>>> tile(b,[2,3])
# 	array([[1, 3, 5, 1, 3, 5, 1, 3, 5],
#        [1, 3, 5, 1, 3, 5, 1, 3, 5]])
	diffMat = tile(inX, (dataSetSize,1)) - dataSet	# tile(A,B), duplicate A, B times 
	# matrix sqrt
	sqDiffMat = diffMat**2
	# axis=0 就是普通的相加, 加入axis=1以后就是将一个矩阵的每一行向量相加
	sqDistances = sqDiffMat.sum(axis=1)
	# print "sqDistances: ", sqDistances
	# L2 norm
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()	# return sorted index
	# Voting with lowest k distances
	classCount = {}		# classCount = {'A': cnt, 'B': cnt}
	for i in range(k):
		# min k distance label
		voteIlabel = labels[sortedDistIndicies[i]]
		# dict.get(key, default=None)
		# key -- 字典中要查找的键。
		# default -- 如果指定键的值不存在时，返回该默认值值。
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# Sort dictionary									# sort in 'cnt' and reverse sort
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

if __name__ == "__main__":
	group, labels = createDataSet()
	clas = classify0([1,1], group, labels, 3)
	print clas