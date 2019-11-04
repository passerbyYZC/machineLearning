# encoding:utf-8

import numpy as np
import operator

def createDataSet():
    """创建数据集以及标签
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

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
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(i), reverse=True)
    return sortedClassCount[0][0]

group, labels = createDataSet()
print(classify0([1.0,1.1], group, labels, 2))