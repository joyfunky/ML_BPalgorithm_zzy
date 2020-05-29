import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Iris数据集共有四个特征，输入层维度为4，共有三类Iris，输出层维度为3
#中间先定义一层隐藏层，共有8个神经元

#激活函数Sigmoid
def SigmoidFunc (X,weight):
    Sigout = np.zeros((weight.shape[1],1),np.float)
    for i in range (weight.shape[1]):
        Sigout[i][0] = 1/(1+np.exp(-X[0][i]))
    return Sigout

#初始化输入层到隐层以及隐层到输出层权值矩阵，初始化学习率，最大迭代次数，误差阈值
def InitialPara (dimX,dimH,dimY):
    W12 = 0.04 * np.random.rand(dimX,dimH)
    W23 = 0.04 * np.random.rand(dimH,dimY)
    learning_rate = 0.2
    maxEpochs = 8000
    errorThreshold = 0.1

    return W12,W23,learning_rate,maxEpochs,errorThreshold

def ForwardPropagation (Xin,Weight):
    #Xin是输入的行向量(维度是前一层神经元个数)，Weight是维度为{前一层神经元个数*下一层神经元个数}的矩阵
    netj = np.dot(Xin,Weight)
    Xout = SigmoidFunc(netj,Weight)
    return Xout

def PredictIris (X,W12,W23):
    PredictHidden = ForwardPropagation(X,W12)
    PredictionY = ForwardPropagation(PredictHidden.T,W23).T

    return PredictionY

def BackPropagation (Xtrain,Ytrain,alfa,W12,W23):
    OutputH = ForwardPropagation(Xtrain,W12)
    OutputY = ForwardPropagation(OutputH.T,W23)
    Ecost = np.sum(np.power(abs(Ytrain-OutputY.T),2))/2       #cost function

    deltaYj = np.zeros((W23.shape[1],1))
    for i in range (W23.shape[1]):
        deltaYj[i] = (OutputY[i][0]-Ytrain[0][i])*OutputY[i][0]*(1-OutputY[i][0])

    dW23 = np.dot(OutputH,deltaYj.T)


    deltaHj = np.dot(W23,deltaYj) * OutputH * (1-OutputH)
    dW12 = np.dot(Xtrain.T,deltaHj.T)

    W23 = W23 - alfa * dW23
    W12 = W12 - alfa*dW12

    return W12,W23,Ecost


if __name__ == '__main__':
    iris_data = pd.read_csv('iris.txt')
    iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    iris_data['class'] = iris_data['class'].apply(lambda x: x.split('-')[1])
    shuffledData = np.random.permutation(iris_data.index)
    iris_data = iris_data.iloc[shuffledData].values

    #上面读取Iris数据集，并按行打乱
    rows,cols = iris_data.shape
    print('Number of Rows: '+str(rows)+'\nNumbers of Cols: '+str(cols)+'\n')
    print('-----------------------------')

    #打印整理所得的打乱的Iris数组，iris_data数组共149行，5列
    print('     sepal_length      sepal_width      petal_length      petal_width          class')
    for i in range (50):
        print(str(i)+': ',end="")
        if i < 10:
            print(' ', end="")
        if i < 100:
            print(' ',end='')
        for j in range (cols):
            print('    '+str(iris_data[i][j])+'          ',end="")
        print('\n',end="")

    print('-----------------------------')


    iris_data_new = np.zeros((rows,cols+2))
    #更改标签值
    for i in range (rows):
        iris_data_new[i][0:4] = iris_data[i][0:4]
        if iris_data[i][4] == 'setosa':
            iris_data_new[i][4] = 1
            iris_data_new[i][5] = 0
            iris_data_new[i][6] = 0
        elif iris_data[i][4] == 'versicolor':
            iris_data_new[i][4] = 0
            iris_data_new[i][5] = 1
            iris_data_new[i][6] = 0
        elif iris_data[i][4] == 'virginica':
            iris_data_new[i][4] = 0
            iris_data_new[i][5] = 0
            iris_data_new[i][6] = 1

    print(iris_data_new)
    print('----------------------------')
    Xtrain = np.zeros((110,4),np.float)
    Ytrain = np.zeros((110,3))

    #从此开始BP神经网络分类

    #Iris数据集的第0~109行数据生成训练集
    for i in range (110):
        for j in range (4):
            Xtrain[i][j] = iris_data_new[i][j]

    for i in range (110):
        Ytrain[i] = iris_data_new[i][4:7]

    print('Training 4 feature input: \n'+str(Xtrain))
    print('-----------------------------')
    print('Training refer output: \n'+str(Ytrain))
    print('-----------------------------')

    #Iris数据集的第110~148行生成测试集
    Xtest = np.zeros((rows-110,4),np.float)
    Ytest = np.zeros((rows-110,3))
    for i in range (rows - 110):
        for j in range (4):
            Xtest[i][j] = iris_data_new[i+110][j]

    for i in range (rows - 110):
        Ytest[i] = iris_data_new[i+110][4:7]

    print('Testing 4 feature input: \n' + str(Xtest))
    print('-----------------------------')
    print('Testing refer output: \n' + str(Ytest))
    print('-----------------------------')


    dimX = 4
    dimH = 8
    dimY = 3
    W12, W23, learning_rate, maxEpochs, errorThreshold = InitialPara (dimX,dimH,dimY)

    #使用训练集进行机器学习
    Ecost = np.zeros((maxEpochs,))
    YPrediction = np.zeros((Ytest.shape[0],Ytest.shape[1]))
    Accuracy = np.zeros((maxEpochs,))
    for i in range (maxEpochs):
        AccCount = 0
        for j in range (Xtrain.shape[0]):
            W12, W23, Ecost[i] = BackPropagation(np.array([Xtrain[j]]),np.array([Ytrain[j]]), learning_rate, W12, W23)

        for k in range (Xtest.shape[0]):
            YPrediction[k] = PredictIris(np.array([Xtest[k]]), W12, W23)

            if np.sum(abs(YPrediction[k] - Ytest[k])) <= errorThreshold:
                AccCount = AccCount + 1
        Accuracy[i] = AccCount/(Xtest.shape[0])
        if np.mod(i,500) == 0:
            print('Iteration : '+str(i)+',  The Prediction Accuracy is : '+str(Accuracy[i]))
        if Accuracy[i] == 1.0:
            Accuracy[i+1:maxEpochs] = 1.0
            Ecost[i+1:maxEpochs] = Ecost[i]
            break

    print('------------------------------')
    YPrediction = np.rint(YPrediction)
    for i in range (YPrediction.shape[0]):
        for j in range (YPrediction.shape[1]):
            YPrediction[i][j] = int(YPrediction[i][j])

    iris_prediction = [[] for i in range (YPrediction.shape[0])]
    for i in range (YPrediction.shape[0]):
        if YPrediction[i][0] == 1 and YPrediction[i][1] == 0 and YPrediction[i][2] == 0:
            iris_prediction[i].append("setosa")
        elif YPrediction[i][0] == 0 and YPrediction[i][1] == 1 and YPrediction[i][2] == 0:
            iris_prediction[i].append("versicolor")
        elif YPrediction[i][0] == 0 and YPrediction[i][1] == 0 and YPrediction[i][2] == 1:
            iris_prediction[i].append("virginica")


    print('The real Iris type: \n')
    for i in range (YPrediction.shape[0]):
        print(iris_data[i+110][4])
    print('------------------------------')
    print('The predictive Iris type: \n')
    for i in range (YPrediction.shape[0]):
        print(iris_prediction[i][0])
    print('------------------------------')

    print('Optimizing Weight')
    print(W12)
    print('-----------------------------')
    print(W23)
    print('-----------------------------')
    print('The Predictive Output: \n'+str(YPrediction))
    print('-----------------------------')
    print('The Ideal Output: \n'+str(Ytest))
    print('-----------------------------')

    Epochs = range(maxEpochs)
    plt.figure()
    plt.subplot(121)
    plt.plot(Epochs, Ecost, color='brown', linestyle='-', linewidth='1')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Function Value')
    plt.title('Loss Function')

    plt.subplot(122)
    plt.plot(Epochs, Accuracy, color='brown', linestyle='-', linewidth='1')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Value')
    plt.title('Prediction Accuracy')


    #绘制花瓣、花萼长度的散点图
    count1 = 0
    count2 = 0
    count3 = 0
    plt.figure()
    plt.subplot(121)
    for i in range (110,rows):
        if iris_data[i][4] == 'setosa':
            if count1 == 0:
                plt.scatter(iris_data[i][0],iris_data[i][2],color="red", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(iris_data[i][0], iris_data[i][2], color="red", marker="o")
        elif iris_data[i][4] == 'versicolor':
            if count2 == 0:
                plt.scatter(iris_data[i][0], iris_data[i][2], color="blue", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(iris_data[i][0], iris_data[i][2], color="blue", marker="x")
        elif iris_data[i][4] == 'virginica':
            if count3 == 0:
                plt.scatter(iris_data[i][0], iris_data[i][2], color="green", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(iris_data[i][0], iris_data[i][2], color="green", marker="^")

    plt.title("Expect Output")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")


    count1 = 0
    count2 = 0
    count3 = 0
    plt.subplot(122)
    for i in range (rows - 110):
        if iris_prediction[i][0] == 'setosa':
            if count1 == 0:
                plt.scatter(Xtest[i][0],Xtest[i][2],color="red", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="red", marker="o")
        elif iris_prediction[i][0] == 'versicolor':
            if count2 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="blue", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="blue", marker="x")
        elif iris_prediction[i][0] == 'virginica':
            if count3 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="green", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="green", marker="^")

    plt.title("Predictive Output")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")


    #绘制花瓣、花萼宽度的散点图
    count1 = 0
    count2 = 0
    count3 = 0
    plt.figure()
    plt.subplot(121)
    for i in range(110, rows):
        if iris_data[i][4] == 'setosa':
            if count1 == 0:
                plt.scatter(iris_data[i][1], iris_data[i][3], color="coral", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(iris_data[i][1], iris_data[i][3], color="coral", marker="o")
        elif iris_data[i][4] == 'versicolor':
            if count2 == 0:
                plt.scatter(iris_data[i][1], iris_data[i][3], color="darkgreen", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(iris_data[i][1], iris_data[i][3], color="darkgreen", marker="x")
        elif iris_data[i][4] == 'virginica':
            if count3 == 0:
                plt.scatter(iris_data[i][1], iris_data[i][3], color="brown", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(iris_data[i][1], iris_data[i][3], color="brown", marker="^")

    plt.title("Expect Output")
    plt.xlabel("sepal width")
    plt.ylabel("petal width")
    plt.legend(loc="upper left")


    count1 = 0
    count2 = 0
    count3 = 0
    plt.subplot(122)
    for i in range(rows - 110):
        if iris_prediction[i][0] == 'setosa':
            if count1 == 0:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="coral", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="coral", marker="o")
        elif iris_prediction[i][0] == 'versicolor':
            if count2 == 0:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="darkgreen", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="darkgreen", marker="x")
        elif iris_prediction[i][0] == 'virginica':
            if count3 == 0:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="brown", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="brown", marker="^")

    plt.title("Predictive Output")
    plt.xlabel("sepal width")
    plt.ylabel("petal width")
    plt.legend(loc="upper left")

    plt.show()



