from numpy import *
import operator
from os import listdir
import kNN

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []	# list
    trainingFileList = listdir('digits/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # fileNameStr = 0_1.txt; 
        # fileStr = 0_1
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        # classNumStr = 0
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # row i
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    # test
    testFileList = listdir('digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

if __name__ == "__main__":
	test_vector = img2vector('digits/testDigits/0_1.txt')
	print test_vector[0, 0:31]
	# hand writing test
	handwritingClassTest()