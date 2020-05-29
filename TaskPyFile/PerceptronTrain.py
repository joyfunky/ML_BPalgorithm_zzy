import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def initialPara (dimX):
    learning_rate = 0.001
    maxEpochs = 4000
    weight = 0.004 * np.random.randn(dimX,1)          #权重矩阵是一个输入同维度的列向量
    biase = 0.004

    return learning_rate,maxEpochs,weight,biase

def initialParaSecond (dimX):
    learning_rate = 0.2
    maxEpochs = 8000
    weight = 0.04 * np.random.randn(dimX,1)          #权重矩阵是一个输入同维度的列向量
    biase = 0.04

    return learning_rate,maxEpochs,weight,biase

def forwardCal (weight,biase,Xin):
    yout = np.dot(Xin,weight) + biase
    if yout >= 0:
        yout = 1
    else:
        yout = -1

    return yout

def RenewPara (weight,biase,x_train,y_train,learning_rate):
    y_output = forwardCal(weight,biase,x_train)
    LossFunc = -((np.dot(x_train,weight)+biase)*y_train)
    deltaWeight = np.zeros((len(x_train),1))
    for i in range (len(x_train)):
        deltaWeight[i] = learning_rate*(y_train-y_output)*x_train[i]
    deltaBiase = learning_rate*y_output
    weight = weight + deltaWeight
    biase = biase + deltaBiase

    return weight,biase,LossFunc



#主函数开始运行
if __name__ == '__main__':
    iris_data = pd.read_csv('iris.txt')
    iris_data.columns = ['sepal length','sepal width','petal length','petal width','class']
    iris_data['class'] = iris_data['class'].apply(lambda x:x.split('-')[1])

    #感知机Perceptron只能实现二分类，需要先将setosa和另两种Iris分开，再对另两种进行二分类
    #先将数据集的versicolor和viginica两类归并成一类
    iris_data_one = iris_data.copy()
    # 打乱生成第一次将setosa和其他iris二分类的数据集iris_data_one
    shuffledData = np.random.permutation(iris_data_one.index)
    iris_data_one = iris_data_one.iloc[shuffledData].values

    for i in range (len(iris_data_one)):
        if iris_data_one[i][4] == "setosa":
            iris_data_one[i][4] = "setosa"
        elif iris_data_one[i][4] == "versicolor" or iris_data_one[i][4] == "virginica":
            iris_data_one[i][4] = "otheriris"

    print('The new dataset : ')
    print(iris_data_one)
    print('-------------------------------')
    iris_data_one_dataframe = iris_data_one


    for i in range (len(iris_data_one)):
        if iris_data_one[i][4] == "setosa":
            iris_data_one[i][4] = 1
        elif iris_data_one[i][4] == "otheriris":
            iris_data_one[i][4] = -1

    print('shuffled Dataset for first classification: \n' + str(iris_data_one))
    print('-------------------------------')

    #划分欲分类setosa和其他iris种类的训练集和测试集
    Xtrain = []
    Ytrain = []
    for i in range (110):
        Xtrain.append(iris_data_one[i][0:4])
        Ytrain.append(iris_data_one[i][4])
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    print('first input training data: \n'+str(Xtrain))
    print('-------------------------------')
    print('first output training data: \n'+str(Ytrain))
    print('-------------------------------')
    Xtest = []
    Ytest = []
    for i in range (len(iris_data_one)):
        Xtest.append(iris_data_one[i][0:4])
        Ytest.append(iris_data_one[i][4])
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    print('first input testing data: \n' + str(Xtest))
    print('-------------------------------')
    print('first output testing data: \n' + str(Ytest))
    print('-------------------------------')



    print('First Training Iteration')
    #开始Preceptron感知机训练和预测,开始第一轮训练，区分setosa和其它品种
    dimX = 4
    learning_rate, maxEpochs, weight, biase = initialPara (dimX)
    Yprediction = np.zeros((len(Ytest),))
    Accuracy = np.zeros((maxEpochs,))
    LossFunc = np.zeros((maxEpochs,))
    for i in range (maxEpochs):
        correctCount = 0
        for j in range (Xtrain.shape[0]):
            weight, biase, LossFunc[i] = RenewPara (weight,biase,Xtrain[j],Ytrain[j],learning_rate)
        for k in range (Xtest.shape[0]):
            Yprediction[k] = forwardCal (weight,biase,Xtest[k])
            if Yprediction[k] == Ytest[k]:
                correctCount = correctCount + 1
        Accuracy[i] = correctCount/Xtest.shape[0]
        if np.mod(i,500) == 0:
            print('Iteration : '+str(i) + ',  Prediction Accuracy : '+str(Accuracy[i]))
        if Accuracy[i] == 1.0:
            Accuracy[i+1:maxEpochs] = 1.0
            LossFunc[i+1:maxEpochs] = LossFunc[i]
            break
    Epochs = range(maxEpochs)
    plt.figure()
    plt.subplot(121)
    plt.plot(Epochs, LossFunc, color='darkorange', linestyle='-', linewidth='1')
    plt.xlabel("Iteration")
    plt.ylabel("Loss Function Value")
    plt.title("First Training Cycle Loss Function")

    plt.subplot(122)
    plt.plot(Epochs, Accuracy, color='brown', linestyle='-', linewidth='1')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Value')
    plt.title('Prediction Accuracy')

    print('Weight Matrix : \n'+str(weight))
    print('-------------------------------')
    print('Biase : \n'+str(biase))
    print('-------------------------------')

    predictionIris = []
    for i in range (Xtest.shape[0]):
        if Yprediction[i] == 1:
            predictionIris.append("setosa")
        else:
            predictionIris.append("otheriris")

    idealIris = []
    for i in range (Xtest.shape[0]):
        if Ytest[i] == 1:
            idealIris.append("setosa")
        else:
            idealIris.append("otheriris")

    print('-------------------------------')
    print('Ideal output is : \n'+str(idealIris))
    print('-------------------------------')
    print('Predictive output is \n'+str(predictionIris))
    print('-------------------------------')

    # 从完整的数据集依据第一次训练结果提取第二次的数据集
    iris_data_three = iris_data.iloc[shuffledData].values

    print('whole shuffled iris datasets :\n' + str(iris_data_three))
    print('-------------------------------')

    Xtrain3 = []
    Ytrain3 = []
    Xtest2 = []
    Ytest2 = []
    Xtest3 = []
    Ytest3 = []
    XtestRemindIndex = []
    XtestRemindIndex2 = []

    for i in range (len(Yprediction)):
        if predictionIris[i] == "otheriris":
            Xtest2.append(iris_data_three[i][0:4])
            XtestRemindIndex.append(i)
            if iris_data_three[i][4] == "versicolor":
                Ytest2.append(1)
            elif iris_data_three[i][4] == "virginica":
                Ytest2.append(-1)
            else:
                Ytest2.append(1)


    for i in range (70):
        Xtrain3.append(Xtest2[i])
        Ytrain3.append(Ytest2[i])

    for i in range (len(Xtest2)):
        Xtest3.append(Xtest2[i])
        Ytest3.append(Ytest2[i])
        XtestRemindIndex2.append(XtestRemindIndex[i])

    Xtest3 = np.array(Xtest3)
    Ytest3 = np.array(Ytest3)
    Xtrain3 = np.array(Xtrain3)
    Ytrain3 = np.array(Ytrain3)

    print('Second Dataset length : '+str(Xtest3.shape[0]))
    print('-------------------------------')
    print('Second Testing datasets')
    print(Xtest3)
    print('-------------------------------')
    print(Ytest3)
    print('-------------------------------')


    print('Second Training Iteration')
    print('-------------------------------')
    # 开始Preceptron感知机训练和预测,开始第二轮训练，区分versicolor和virginica
    learning_rate, maxEpochs, weight, biase = initialParaSecond (dimX)
    Yprediction2 = np.zeros((len(Ytest3),),np.int)
    Accuracy2 = np.zeros((maxEpochs,))
    LossFunc2 = np.zeros((maxEpochs,))
    for i in range(maxEpochs):
        correctCount = 0
        for j in range(Xtrain3.shape[0]):
            weight, biase, LossFunc2[i] = RenewPara(weight, biase, Xtrain3[j], Ytrain3[j], learning_rate)
        for k in range(Xtest3.shape[0]):
            Yprediction2[k] = np.round(forwardCal(weight, biase, Xtest3[k]))
            if Yprediction2[k] == Ytest3[k]:
                correctCount = correctCount + 1
        Accuracy2[i] = correctCount/Xtest3.shape[0]
        if np.mod(i, 500) == 0:
            print('Iteration : ' + str(i) + ',  Prediction Accuracy : ' + str(Accuracy2[i]))
        if Accuracy2[i] == 1.0:
            Accuracy2[i+1:maxEpochs] = 1.0
            LossFunc2[i+1:maxEpochs] = LossFunc2[i]

    Epochs = range(maxEpochs)

    plt.figure()
    plt.subplot(121)
    plt.plot(Epochs, LossFunc2, color='darkgreen', linestyle='-',linewidth = '1')
    plt.xlabel("Iteration")
    plt.ylabel("Loss Function Value")
    plt.title("Second Training Cycle Loss Function")

    plt.subplot(122)
    plt.plot(Epochs, Accuracy2, color='brown', linestyle='-', linewidth='1')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Value')
    plt.title('Prediction Accuracy')



    print('Weight Matrix : \n' + str(weight))
    print('-------------------------------')
    print('Biase : \n' + str(biase))
    print('-------------------------------')
    print(str(Yprediction2))
    print('-------------------------------')


    predictionIrisFinal1 = np.zeros((Xtest.shape[0],))
    predictionIrisFinal = []
    idealIrisFinal = []

    for i in range (Xtest.shape[0]):
        if predictionIris[i] == "setosa":
            predictionIrisFinal1[i] = 0

    for i in range (Xtest3.shape[0]):
        if Yprediction2[i] == 1:
            predictionIrisFinal1[XtestRemindIndex[i]] = 1
        elif Yprediction2[i] == -1:
            predictionIrisFinal1[XtestRemindIndex[i]] = 2

    for i in range (Xtest.shape[0]):
        if predictionIrisFinal1[i] == 0:
            predictionIrisFinal.append("setosa")
        elif predictionIrisFinal1[i] == 1:
            predictionIrisFinal.append("versicolor")
        else:
            predictionIrisFinal.append("virginica")

    for i in range (Xtest.shape[0]):
        idealIrisFinal.append(iris_data_three[i][4])

    print('final Iris ideal type : ')
    print(idealIrisFinal)
    print('-------------------------------')
    print('final Iris prediction : ')
    print(predictionIrisFinal)
    print('-------------------------------')


    #绘制第一次分类图
    count1 = 0
    count2 = 0
    count3 = 0
    plt.figure()
    plt.subplot(121)
    for i in range(Xtest3.shape[0]):
        if idealIris[i] == "setosa":
            if count1 == 0:
                plt.scatter(iris_data_one[i][0], iris_data_one[i][2], color="red", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(iris_data_one[i][0], iris_data_one[i][2], color="red", marker="o")
        elif idealIris[i] == "otheriris":
            if count2 == 0:
                plt.scatter(iris_data_one[i][0], iris_data_one[i][2], color="blue", marker="x", label="versicolor or virginica")
                count2 = count2 + 1
            else:
                plt.scatter(iris_data_one[i][0], iris_data_one[i][2], color="blue", marker="x")

    plt.title("First Classification Perceptron ideal Output")
    plt.xlabel("sepal width")
    plt.ylabel("petal width")
    plt.legend(loc="upper left")

    count1 = 0
    count2 = 0
    count3 = 0
    plt.subplot(122)
    for i in range(Xtest3.shape[0]):
        if predictionIris[i] == "setosa":
            if count1 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="red", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="red", marker="o")
        elif predictionIris[i] == "otheriris":
            if count2 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="blue", marker="x", label="versicolor or virginica")
                count2 = count2 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="blue", marker="x")

    plt.title("First Classification Perceptron predictive Output")
    plt.xlabel("sepal width")
    plt.ylabel("petal width")
    plt.legend(loc="upper left")


    #绘制第二次分类图
    count1 = 0
    count2 = 0
    count3 = 0
    plt.figure()
    plt.subplot(121)
    for i in range (Xtest3.shape[0]):
        if idealIrisFinal[i] == "setosa":
            if count1 == 0:
                plt.scatter(iris_data_one[i][0], iris_data_one[i][2], color="red", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(iris_data_one[i][0], iris_data_one[i][2], color="red", marker="o")
        elif idealIrisFinal[i] == "versicolor":
            if count2 == 0:
                plt.scatter(iris_data_one[i][0],iris_data_one[i][2], color="blue", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(iris_data_one[i][0], iris_data_one[i][2], color="blue", marker="x")
        elif idealIrisFinal[i] == "virginica":
            if count3 == 0:
                plt.scatter(iris_data_one[i][0],iris_data_one[i][2], color="green", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(iris_data_one[i][0], iris_data_one[i][2], color="green", marker="^")
    plt.title("Final Classification Perceptron ideal Output")
    plt.xlabel("sepal width")
    plt.ylabel("petal width")
    plt.legend(loc="upper left")


    count1 = 0
    count2 = 0
    count3 = 0
    plt.subplot(122)
    for i in range(Xtest3.shape[0]):
        if predictionIrisFinal[i] == "setosa":
            if count1 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="red", marker="o", label="setosa")
                count1 = count1 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="red", marker="o")
        elif predictionIrisFinal[i] == "versicolor":
            if count2 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="blue", marker="x", label="versicolor")
                count2 = count2 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="blue", marker="x")
        elif predictionIrisFinal[i] == "virginica":
            if count3 == 0:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="green", marker="^", label="virginica")
                count3 = count3 + 1
            else:
                plt.scatter(Xtest[i][0], Xtest[i][2], color="green", marker="^")
    plt.title("Final Classification Perceptron predictive Output")
    plt.xlabel("sepal width")
    plt.ylabel("petal width")
    plt.legend(loc="upper left")

    plt.show()

