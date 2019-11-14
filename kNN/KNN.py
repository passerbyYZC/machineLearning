# encoding:utf-8

import os
import numpy as np
import operator


def classify0(inX, dataSet, labels, k):
    """样本分类

    Args:
        inX: 用于分类的输入向量
        dataSet: 训练样本集
        labels: 标签向量
        k: 选择最近邻居数目
    
    Return:
        输入向量所属类别
    """

    dataSetSize = dataSet.shape[0]

    # 距离计算并升序排序
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSetSize
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5    
    sortedDistIndicies = distances.argsort()

    # 计算最近临近类别频率并选择最近的k个邻居
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    """读取文本数据并结构化

    Args:
        filename: 文本文件路径
    
    Return:
        数据样本矩阵、样本标签向量
    """
    with open(filename, "r") as fr:
        arrayOlines = fr.readlines()
        numberOfLines = len(arrayOlines)  # 样本数

    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 解析数据
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """归一化特征值

    Args:
        dataSet: 样本数据集

    Return:
        归一化样本数据集、范围宽带、最小值
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, range, minVals

def img2vector(filename):
    """图像转换为向量
    
    Args:
        filename: 文本文件路径
    
    Return:
        图像向量
    """
    returnVect = np.zeros((1,1024))
    with open(filename, "r") as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest(trainingFilesDir, testFilsDir):
    """手写数字识别系统测试函数
    Args:
        trainingFilesDir: 训练文件目录
        testFilsDir: 测试文件目录
    """
    hwLabels = []
    trainingFileList = os.listdir(trainingFilesDir)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in  range(m):
        # 获取训练图像标签
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        # 获取训练图像数据
        trainingMat[i,:] = img2vector(trainingFilesDir + '\\' + fileNameStr)

    # 获取测试数据
    testFileList = os.listdir(testFilsDir)
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        # 获取测试图像标签
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        # 获取测试图像数据
        vectorUnderTest = img2vector(testFilsDir + '\\' + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if(classifierResult != classNumStr): 
            errorCount += 1
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
