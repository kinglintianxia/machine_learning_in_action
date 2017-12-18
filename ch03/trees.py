#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
import operator
import copy
# 
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']   # 不浮出水面, 脚蹼
    #change to discrete values
    return dataSet, labels
# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)   # 
    labelCounts = {}    # 
    for featVec in dataSet: #the number of unique elements and their occurance
        currentLabel = featVec[-1]  # lable: yes or no
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
# 划分数据集
def splitDataSet(dataSet, axis, value): # ：数据集、划分数据集的特征、需要返回的特征的值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:]) # [1,2,3+ 4,5,6]
            retDataSet.append(reducedFeatVec)   # [[#1],[#2]]
    return retDataSet
# 熵计算将会告诉我们如何划分数据集是最好的数据组织方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)   # the origin entropy
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        # i = 0, featList = [1,1,1,0,0]
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # i = 0, uniqueVals = [1,0]
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer
# 多数表决的方法
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel
# 决策树的存储
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
if __name__ == "__main__":
    # 1
    dataSet, labels = createDataSet()
    labels_copy = copy.deepcopy(labels)
    # 
    # dataSet[0][-1] = "maybe"
    print "dataSet: ", dataSet
    print "lables: ", labels
    # 
    shannonEnt = calcShannonEnt(dataSet)
    print "shannonEnt: ", shannonEnt
    # 
    print "split(0,1): ", splitDataSet(dataSet, 0, 1)
    print "split(0,0): ", splitDataSet(dataSet, 1, 0)
    print "best Feature: ", chooseBestFeatureToSplit(dataSet)
    # classify
    tree = createTree(dataSet, labels)
    print "tree: ", tree
    print "classify [1,0]: ", classify(tree, labels_copy, [1,0])
    print "classify [1,1]:", classify(tree, labels_copy, [1,1])
    # store tree
    storeTree(tree, 'DesitionTree.txt')
    print grabTree('DesitionTree.txt')
