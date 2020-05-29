import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#初始化函数，设置BP神经网络含有一个输入层，一个隐藏层，一个输出层
def InitialPara (dimX,dimY,dimH):
    learning_rate = 0.003
    w12 = 0.004*np.random.randn(dimX,dimH)
    w23 = 0.008*np.random.randn(dimH,dimY)
    errorThreshold = 0.5
    maxEpochs = 30000

    return learning_rate,w12,w23,errorThreshold,maxEpochs

#Sigmoid激活函数，设定输入为列向量矩阵，维度为某层神经元个数
def SigmoidActivation (xin):
    yout = np.zeros((xin.shape[0],1))
    for i in range (xin.shape[0]):
        yout[i][0] = 1/(1+np.exp(-xin[i][0]))
    return yout

def forwardPropagation (xinput,weight):
    #注意xinput应当为1*n维数组
    yout = np.transpose(np.dot(xinput,weight))

    return yout

def CalOutput (xinput,w12,w23):
    HiddenOut = forwardPropagation(xinput,w12)
    HiddenOut = SigmoidActivation(HiddenOut)
    yout = forwardPropagation(np.transpose(HiddenOut),w23)

    return yout

def BackPropagation (xtrain,ytrain,w12,w23,learning_rate):
    deltaYj = np.zeros((w23.shape[1],1))
    deltaHj = np.zeros((w12.shape[1],1))

    HiddenOut = forwardPropagation(xtrain,w12)
    HiddenOut = SigmoidActivation(HiddenOut)
    yout = forwardPropagation(np.transpose(HiddenOut),w23)
    Lossfunc = 0.5*np.sum(np.power(abs(ytrain-yout),2))

    deltaYj = yout - ytrain
    deltaHj = (np.dot(w23,deltaYj))*HiddenOut*(1-HiddenOut)
    w23 = w23 - learning_rate*(np.dot(HiddenOut,np.transpose(deltaYj)))
    w12 = w12 - learning_rate*(np.dot(np.transpose(xtrain),np.transpose(deltaHj)))

    return w12,w23,Lossfunc


#主程序入口
if __name__ == '__main__':
    #autoencoder的输入层选取Iris数据集的四个输入量，输出层则同样为四个输出量，用BP神经网络让输出逼近输入
    readData = pd.read_csv('iris.txt')
    readData.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    readData['class'] = readData['class'].apply(lambda x: x.split('-')[1])

    readData_one = readData.copy()
    shuffleData = np.random.permutation(readData_one.index)
    readData_one = readData_one.iloc[shuffleData].values

    print('Shuffled Data : \n'+str(readData_one))
    print('-------------------------------')

    #得到预处理后的数据集，输入输出均为二维list形式
    DataInput = []
    DataOutput = []
    for i in range (readData_one.shape[0]):
        DataInput.append(list(readData_one[i][0:4]))
        DataOutput.append(list(readData_one[i][0:4]))

    print('Input Datasets : \n'+str(DataInput))
    print('-------------------------------')
    print('Output Datasets: \n'+str(DataOutput))
    print('-------------------------------')


    Xtrain = []
    Ytrain = []
    for i in range (110):
        Xtrain.append(readData_one[i][0:4])
        Ytrain.append(readData_one[i][0:4])
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtest = []
    Ytest = []
    for i in range (len(readData_one)-110):
        Xtest.append(readData_one[i+110][0:4])
        Ytest.append(readData_one[i+110][0:4])

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)

    print('The training input dataset : \n'+str(Xtrain))
    print('-------------------------------')
    print('The training output dataset : \n'+str(Ytrain))
    print('-------------------------------')
    print('The testing input dataset : \n'+str(Xtest))
    print('-------------------------------')
    print('The testing output dataset : \n'+str(Ytest))
    print('-------------------------------')

    dimX = 4
    dimY = 4
    dimH = 10
    learning_rate, w12, w23, errorThreshold, maxEpochs = InitialPara (dimX,dimY,dimH)
    Accuracy = np.zeros((maxEpochs,))
    Yprediction = np.zeros((Xtest.shape[0],dimY))
    LossFunc = np.zeros((maxEpochs,))

    for i in range (maxEpochs):
        AccCount = 0
        for j in range (Xtrain.shape[0]):
            w12,w23,LossFunc[i] = BackPropagation (np.array([Xtrain[j]]),np.transpose(np.array([Ytrain[j]])),w12,w23,learning_rate)

        for k in range (Xtest.shape[0]):
            Yprediction[k] = np.transpose(CalOutput(np.array([Xtest[k]]),w12,w23))
            if np.sum(abs(Yprediction[k] - Ytest[k])) < errorThreshold:
                AccCount = AccCount + 1
        Accuracy[i] = AccCount/(Xtest.shape[0])
        if np.mod(i,500) == 0:
            print('Iteration :'+str(i)+', prediction accuracy : '+str(Accuracy[i]))

    print('-------------------------------')
    print('Ideal output : \n'+str(Ytest))
    print('-------------------------------')
    print('Predictive output : \n'+str(Yprediction))

    Epochs = range(maxEpochs)
    plt.figure()
    plt.subplot(121)
    plt.plot(Epochs,LossFunc,color='brown',linestyle = '-',linewidth = '1')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Function Value')
    plt.title('Loss Function')

    plt.subplot(122)
    plt.plot(Epochs, Accuracy, color='brown', linestyle='-', linewidth='1')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Value')
    plt.title('Prediction Accuracy')

    count = 0
    plt.figure()
    for i in range (Ytest.shape[0]):
        if count == 0:
            plt.scatter(Ytest[i][0], Ytest[i][2], color="red", marker="o", label="Expect Output")
            plt.scatter(Yprediction[i][0], Yprediction[i][2], color="brown", marker="x", label="Predictive Output")
            count = count + 1
        else:
            plt.scatter(Ytest[i][0], Ytest[i][2], color="red", marker="o")
            plt.scatter(Yprediction[i][0], Yprediction[i][2], color="brown", marker="x")

    plt.title("sepal and petal length output")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")

    count = 0
    plt.figure()
    for i in range (Ytest.shape[0]):
        if count == 0:
            plt.scatter(Ytest[i][1], Ytest[i][3], color="blue", marker="o", label="Expect Output")
            plt.scatter(Yprediction[i][1], Yprediction[i][3], color="green", marker="x", label="Predictive Output")
            count = count + 1
        else:
            plt.scatter(Ytest[i][1], Ytest[i][3], color="blue", marker="o")
            plt.scatter(Yprediction[i][1], Yprediction[i][3], color="green", marker="x")

    plt.title("sepal and petal width output")
    plt.xlabel("sepal width")
    plt.ylabel("petal width")
    plt.legend(loc="upper left")

    plt.show()


