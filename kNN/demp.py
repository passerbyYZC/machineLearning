import KNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import array


# filePath = r"C:\Users\Yaozh\Documents\VSCode\github\machineLearning\machineLearning\kNN\datingTestSet2.txt"
# data, labels = KNN.file2matrix(filePath)
# normMat, range, minVals = KNN.autoNorm(data)
# data = normMat
# fig = plt.figure()
# ax = fig.add_subplot(131)
# ax.scatter(data[:, 1], data[:, 2], 15.0*array(labels), 15.0*array(labels))
# ax = fig.add_subplot(132)
# ax.scatter(data[:, 0], data[:, 2], 15.0*array(labels), 15.0*array(labels))
# ax = fig.add_subplot(133)
# ax.scatter(data[:, 0], data[:, 1], 15.0*array(labels), 15.0*array(labels))
# plt.show()


trainingFilesDir = r"C:\Users\Yaozh\Documents\VSCode\github\machineLearning\machineLearning\kNN\digits\trainingDigits"
testFilesDir = r"C:\Users\Yaozh\Documents\VSCode\github\machineLearning\machineLearning\kNN\digits\testDigits"


KNN.handwritingClassTest(trainingFilesDir, testFilesDir)
