import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#两层隐藏层BP网络初始化
def InitializeTwolayer (dimX,dimH1,dimH2,dimY):
    learning_rate = 0.03
    w12 = 0.02*np.random.rand(dimX,dimH1)
    w23 = 0.02*np.random.rand(dimH1,dimH2)
    w34 = 0.04*np.random.rand(dimH2,dimY)
    maxEpochs = 8000
    errorThreshold = 0.1
    ActivationFunc = "relu"

    return learning_rate,maxEpochs,errorThreshold,w12,w23,w34,ActivationFunc

#sigmoid激活函数
def sigmoidActivation (xinput):
    #函数的输入为某层神经元的输出列向量
    yout = np.zeros((xinput.shape[0],1))
    for i in range (xinput.shape[0]):
        yout[i][0] = 1/(1+np.exp(-xinput[i][0]))

    return yout

#tanh激活函数
def tanhActivation (xinput):
    yout = np.zeros((xinput.shape[0], 1))
    for i in range (xinput.shape[0]):
        yout[i][0] = (np.exp(xinput[i][0])-np.exp(-xinput[i][0]))/(np.exp(xinput[i][0])+np.exp(-xinput[i][0]))

    return yout

#relu激活函数
def reluActivation (xinput):
    yout = np.zeros((xinput.shape[0], 1))
    for i in range (xinput.shape[0]):
        if xinput[i][0]>=0:
            yout[i][0] = xinput[i][0]
        else:
            yout[i][0] = 0
    return yout

def forwardPropagation (xinput,weight):
    yout = np.transpose(np.dot(xinput,weight))
    return yout

#前向传播预测输出
def PredictOutput (xinput,w12,w23,w34,ActivationFunc):
    if ActivationFunc == "sigmoid":
        Hidden1Out = sigmoidActivation(forwardPropagation(xinput,w12))
        Hidden2Out = sigmoidActivation(forwardPropagation(np.transpose(Hidden1Out),w23))
        yout = sigmoidActivation(forwardPropagation(np.transpose(Hidden2Out),w34))
    elif ActivationFunc == "tanh":
        Hidden1Out = tanhActivation(forwardPropagation(xinput, w12))
        Hidden2Out = tanhActivation(forwardPropagation(np.transpose(Hidden1Out), w23))
        yout = tanhActivation(forwardPropagation(np.transpose(Hidden2Out), w34))
    elif ActivationFunc == "relu":
        Hidden1Out = reluActivation(forwardPropagation(xinput, w12))
        Hidden2Out = sigmoidActivation(forwardPropagation(np.transpose(Hidden1Out), w23))
        yout = reluActivation(forwardPropagation(np.transpose(Hidden2Out), w34))
    return yout

#反向传播计算误差并更新权值
def BackPropagation (xtrain,ytrain,w12,w23,w34,learning_rate,ActivationFunc):
    #输入的xtrain为1*n矩阵，ytrain为n*1矩阵
    if ActivationFunc == "sigmoid":
        Hidden1Out = sigmoidActivation(forwardPropagation(xtrain,w12))
        Hidden2Out = sigmoidActivation(forwardPropagation(np.transpose(Hidden1Out),w23))
        yout = sigmoidActivation(forwardPropagation(np.transpose(Hidden2Out),w34))
        deltaY = (yout-ytrain)*yout*(1-yout)
        deltaH2 = (np.dot(w34,deltaY))*Hidden2Out*(1-Hidden2Out)
        deltaH1 = (np.dot(w23,deltaH2))*Hidden1Out*(1-Hidden1Out)
        w12 = w12-learning_rate*np.dot(np.transpose(xtrain),np.transpose(deltaH1))
        w23 = w23-learning_rate*np.dot(Hidden1Out,np.transpose(deltaH2))
        w34 = w34-learning_rate*np.dot(Hidden2Out,np.transpose(deltaY))
    elif ActivationFunc == "tanh":
        Hidden1Out = tanhActivation(forwardPropagation(xtrain, w12))
        Hidden2Out = tanhActivation(forwardPropagation(np.transpose(Hidden1Out), w23))
        yout = tanhActivation(forwardPropagation(np.transpose(Hidden2Out), w34))
        deltaY = (yout - ytrain) * (1-np.power(yout,2))
        deltaH2 = (np.dot(w34, deltaY)) * (1-np.power(Hidden2Out,2))
        deltaH1 = (np.dot(w23, deltaH2)) * (1-np.power(Hidden1Out,2))
        w12 = w12 - learning_rate * np.dot(np.transpose(xtrain), np.transpose(deltaH1))
        w23 = w23 - learning_rate * np.dot(Hidden1Out, np.transpose(deltaH2))
        w34 = w34 - learning_rate * np.dot(Hidden2Out, np.transpose(deltaY))
    elif ActivationFunc == "relu":
        #两个隐层和一个输出层都使用relu激活最高只能到64.10%，此函数尝试relu-sigmoid-relu的激活配置
        Hidden1 = forwardPropagation(xtrain, w12)
        Hidden1Out = reluActivation(Hidden1)
        Hidden2 = forwardPropagation(np.transpose(Hidden1Out), w23)
        Hidden2Out = sigmoidActivation(Hidden2)
        y = forwardPropagation(np.transpose(Hidden2Out), w34)
        yout = reluActivation(y)

        deltaY = np.zeros((yout.shape[0],1))
        deltaH2 = np.zeros((Hidden2Out.shape[0],1))
        deltaH1 = np.zeros((Hidden1Out.shape[0],1))

        for i in range (deltaY.shape[0]):
            if y[i][0]>=0:
                deltaY[i][0] = 1*(yout[i][0]-ytrain[i][0])
            else:
                deltaY[i][0] = 0

        deltaH2 = (np.dot(w34, deltaY)) * Hidden2Out * (1 - Hidden2Out)

        for i in range (deltaH1.shape[0]):
            if Hidden1[i][0]>=0:
                deltaH1[i][0] = 1*(np.dot(w23[i],deltaH2))
            else:
                deltaH1[i][0] = 0


        w12 = w12 - learning_rate * np.dot(np.transpose(xtrain), np.transpose(deltaH1))
        w23 = w23 - learning_rate * np.dot(Hidden1Out, np.transpose(deltaH2))
        w34 = w34 - learning_rate * np.dot(Hidden2Out, np.transpose(deltaY))

    Ecost = 0.5*np.sum(np.power(abs(yout-ytrain),2))

    return w12,w23,w34,Ecost




if __name__ == '__main__':
    datasets = pd.read_csv('iris.txt')
    datasets.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    datasets['class'] = datasets['class'].apply(lambda x:x.split('-')[1])
    shuffledData = np.random.permutation(datasets.index)
    datasets = datasets.iloc[shuffledData].values

    print('-----------------------')
    for i in range (len(datasets)):
        print(datasets[i][4])

    irisData = []
    for i in range (datasets.shape[0]):
        irisData.append([])
        for j in range (datasets.shape[1]-1):
            irisData[i].append(datasets[i][j])
        if datasets[i][4] == 'setosa':
            irisData[i].append(1)
            irisData[i].append(0)
            irisData[i].append(0)
        elif datasets[i][4] == 'versicolor':
            irisData[i].append(0)
            irisData[i].append(1)
            irisData[i].append(0)
        elif datasets[i][4] == 'virginica':
            irisData[i].append(0)
            irisData[i].append(0)
            irisData[i].append(1)

    irisData = np.array(irisData)

    print('-----------------------')
    print('Shuffled datasets : \n' + str(irisData))
    print('-----------------------')

    Xtrain = []
    Ytrain = []
    for i in range (110):
        Xtrain.append(irisData[i][0:4])
        Ytrain.append(irisData[i][4:7])

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtest = []
    Ytest = []
    for i in range (irisData.shape[0]-110):
        Xtest.append(irisData[i+110][0:4])
        Ytest.append(irisData[i+110][4:7])

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)

    print('Training iris datasets input : \n'+str(Xtrain))
    print('-----------------------')
    print('Training iris datasets output : \n'+str(Ytrain))
    print('-----------------------')
    print('Testing iris datasets input : \n'+str(Xtest))
    print('-----------------------')
    print('Testing iris datasets output : \n'+str(Ytest))
    print('-----------------------')

    dimX = 4
    dimY = 3
    dimH1 = 5
    dimH2 = 4

    learning_rate, maxEpochs, errorThreshold, w12, w23, w34, ActivationFunc = InitializeTwolayer (dimX,dimH1,dimH2,dimY)
    Accuracy = np.zeros((maxEpochs,))
    Yprediction = np.zeros((Ytest.shape[0],dimY))
    Ecost = np.zeros((maxEpochs,))

    for i in range (maxEpochs):
        AccCount = 0
        for j in range (Xtrain.shape[0]):
            w12,w23,w34,Ecost[i] = BackPropagation(np.array([Xtrain[j]]),np.transpose(np.array([Ytrain[j]])),w12,w23,w34,learning_rate,ActivationFunc)
        for k in range (Xtest.shape[0]):
            Yprediction[k] = np.transpose(PredictOutput(np.array([Xtest[k]]),w12,w23,w34,ActivationFunc))
            if np.sum(abs(Yprediction[k]-Ytest[k])) < errorThreshold:
                AccCount = AccCount + 1
        Accuracy[i] = AccCount/Xtest.shape[0]
        if np.mod(i,500) == 0:
            print('Iteration : '+str(i)+', prediction accuracy : '+str(Accuracy[i]))
        if (Accuracy[i] == 1.0):
            Accuracy[i+1:maxEpochs] = 1.0
            Ecost[i+1:maxEpochs] = Ecost[i]
            break

    Yprediction = np.rint(Yprediction)
    for i in range (Yprediction.shape[0]):
        for j in range (Yprediction.shape[1]):
            Yprediction[i][j] = int(Yprediction[i][j])

    print('-----------------------')
    print('The ideal testing output :\n'+str(Ytest))
    print('-----------------------')
    print('The predictive testing output :\n'+str(Yprediction))
    print('-----------------------')

    Epochs = range(maxEpochs)
    plt.figure()
    plt.subplot(121)
    plt.plot(Epochs,Ecost,'-',color='darkgreen',linewidth=1.0)
    plt.xlabel('Iteration')
    plt.ylabel('Error Value')
    plt.title('Cost Function')

    plt.subplot(122)
    plt.plot(Epochs, Accuracy, '-', color='purple', linewidth=1.0)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Value')
    plt.title('Prediction Accuracy')

    irisPrediction = [[] for i in range (Yprediction.shape[0])]
    for i in range (Yprediction.shape[0]):
        if Yprediction[i][0] == 1. and Yprediction[i][1] == 0. and Yprediction[i][2] == 0.:
            irisPrediction[i].append("setosa")
        elif Yprediction[i][0] == 0. and Yprediction[i][1] == 1. and Yprediction[i][2] == 0.:
            irisPrediction[i].append("versicolor")
        elif Yprediction[i][0] == 0. and Yprediction[i][1] == 0. and Yprediction[i][2] == 1.:
            irisPrediction[i].append("virginica")

    print('-----------------------')

    print('The ideal iris type : ')
    for i in range (Yprediction.shape[0]):
        print(datasets[i+110][4])
    print('-----------------------')
    print('The predictive iris type : ')
    for i in range (Yprediction.shape[0]):
        print(irisPrediction[i][0])

    #绘制花瓣花萼长度的散点图
    count1 = 0
    count2 = 0
    count3 = 0
    plt.figure()
    plt.subplot(121)
    for i in range(Yprediction.shape[0]):
        if datasets[i+110][4] == 'setosa':
            if count1 == 0:
                plt.scatter(datasets[i+110][0], datasets[i+110][2], color="red", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(datasets[i+110][0], datasets[i+110][2], color="red", marker="o")
        elif datasets[i+110][4] == 'versicolor':
            if count2 == 0:
                plt.scatter(datasets[i+110][0], datasets[i+110][2], color="blue", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(datasets[i+110][0], datasets[i+110][2], color="blue", marker="x")
        elif datasets[i+110][4] == 'virginica':
            if count3 == 0:
                plt.scatter(datasets[i+110][0], datasets[i+110][2], color="green", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(datasets[i+110][0], datasets[i+110][2], color="green", marker="^")

    plt.title("Expect Output")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")

    count1 = 0
    count2 = 0
    count3 = 0
    plt.subplot(122)
    for i in range(Yprediction.shape[0]):
        if irisPrediction[i][0] == 'setosa':
            if count1 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="red", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="red", marker="o")
        elif irisPrediction[i][0] == 'versicolor':
            if count2 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="blue", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="blue", marker="x")
        elif irisPrediction[i][0] == 'virginica':
            if count3 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="green", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="green", marker="^")

    plt.title("Predictive Output")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")

    #绘制花瓣、花萼的宽度散点图
    count1 = 0
    count2 = 0
    count3 = 0
    plt.figure()
    plt.subplot(121)
    for i in range(Yprediction.shape[0]):
        if datasets[i + 110][4] == 'setosa':
            if count1 == 0:
                plt.scatter(datasets[i+110][1], datasets[i+110][3], color="darkred", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(datasets[i+110][1], datasets[i+110][3], color="darkred", marker="o")
        elif datasets[i+110][4] == 'versicolor':
            if count2 == 0:
                plt.scatter(datasets[i+110][1], datasets[i+110][3], color="darkcyan", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(datasets[i+110][1], datasets[i+110][3], color="darkcyan", marker="x")
        elif datasets[i+110][4] == 'virginica':
            if count3 == 0:
                plt.scatter(datasets[i+110][1], datasets[i+110][3], color="olive", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(datasets[i+110][1], datasets[i+110][3], color="olive", marker="^")

    plt.title("Expect Output")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")

    count1 = 0
    count2 = 0
    count3 = 0
    plt.subplot(122)
    for i in range(Yprediction.shape[0]):
        if irisPrediction[i][0] == 'setosa':
            if count1 == 0:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="darkred", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="darkred", marker="o")
        elif irisPrediction[i][0] == 'versicolor':
            if count2 == 0:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="darkcyan", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="darkcyan", marker="x")
        elif irisPrediction[i][0] == 'virginica':
            if count3 == 0:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="olive", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(Xtest[i][1], Xtest[i][3], color="olive", marker="^")

    plt.title("Predictive Output")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")



    plt.show()

