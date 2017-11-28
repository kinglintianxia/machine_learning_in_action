#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import operator
# user defined
import kNN

def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	# #一维数组: zeros(3) ; 二维数组: zeros((2,3)) 
	returnMat = zeros((numberOfLines, 3))
	classLabelsVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		# Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）。
		line = line.strip()
		listFromLine = line.split('\t')
		# list.insert(index, obj) 将对象插入列表
		# list [index,:]	# row index
		returnMat[index,:] = listFromLine[0:3]
		classLabelsVector.append(int(listFromLine[-1])) # 倒数第1个元素
		index += 1
	return returnMat, classLabelsVector

def plot(datingMat, datingLabels):
	import matplotlib
	import matplotlib.pyplot as plt 
	fig = plt.figure()
	ax = fig.add_subplot(111)	# ax = fig.add_subplot(349), 将画布分割成3行4列，图像画在从左到右从上到下的第9块
	# ax.scatter(datingMat[:,1], datingMat[:,2], size, color)	# 矩阵的第二、第三列数据
	ax.scatter(datingMat[:,0], datingMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
	plt.show()

# Formula:
# normVal = (oldVal-min)/(max-min)
def autoNorm(datingMat):
	minVal = datingMat.min(0)	# colum min; 1*3 matrix		# axis=0; 每列的最小值  
	maxVal = datingMat.max(0)	# colum max; 1*3 matrix		# axis=1；每行的最小值
	ranges = maxVal - minVal
	normDataSet = zeros(shape(datingMat))
	rows = datingMat.shape[0]
	# oldVal-min
	# >>> b=[1,3,5]
	# >>> tile(b,[2,3])
	# array([[1, 3, 5, 1, 3, 5, 1, 3, 5],
 	#       [1, 3, 5, 1, 3, 5, 1, 3, 5]])
	normDataSet = datingMat - tile(minVal,(rows,1))	# rows*3
	# 
	normDataSet = normDataSet/tile(ranges,(rows,1))
	return normDataSet, ranges, minVal

# 测试分类器效果
def datingClassTest():
	ratio = 0.10	# test ratio
	# Sparse data
	datingMat, datingLabels = file2matrix("./datingTestSet2.txt")
	# Norm data
	normMat, ranges, minVal = autoNorm(datingMat)
	# Number test
	rows = normMat.shape[0]
	# first 10%
	numTest = int(rows*ratio)
	errorCnt = 0.0
	for i in range(numTest):
							# rows i 			# train set  								# k = 3
		clas = kNN.classify0(normMat[i,:], normMat[numTest:rows,:], datingLabels[numTest:rows], 5)
		print "The classifier came back with: %d, the real class: %d" % (clas, datingLabels[i])
		if (clas != datingLabels[i]):
			errorCnt += 1.0
	print "The error count: %f" % (errorCnt/float(numTest)) 

# 约会网站预测函数
def classifyPerson():
	resList = ["not at all", 'in small doses', 'in large doses']
	# Input person information
	play = float(raw_input("percentage of time spent playing video games?"))
	fly = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))
	# Reading in file
	# Sparse data
	datingMat, datingLabels = file2matrix("./datingTestSet2.txt")
	# Norm data
	normMat, ranges, minVal = autoNorm(datingMat)
	inArr = array([play, fly, iceCream])
	# normalize inArr
	classify = kNN.classify0((inArr-minVal)/ranges, normMat, datingLabels, 3)
	print "You will probaly like this person: ", resList[classify-1]

if __name__ == "__main__":
	datingMat, datingLabels = file2matrix("./datingTestSet2.txt")
	
	## plot
	# print datingMat[:20], datingLabels[:20]
	# plot(datingMat, datingLabels)
	
	## Normalizing data set
	# normDataSet, ranges, minVal = autoNorm(datingMat)
	# print "normDataSet: \n", normDataSet, '\nranges: \n', ranges, '\nminVal: \n', minVal

	# datingClassTest()
	classifyPerson()