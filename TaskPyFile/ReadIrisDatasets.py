import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Iris数据集0-48行为setosa，49-98行为versicolor，99-148行为virginica

iris_data = pd.read_csv('iris.txt')
iris_data.columns=['sepal_length','sepal_width','petal_length','petal_width','class']
print(iris_data.head(10))
print('--------------------------')
iris_data['class'] = iris_data['class'].apply(lambda x: x.split('-')[1])
print(iris_data[0:len(iris_data):1])
print('--------------------------')

x11 = iris_data.iloc[0:49:1,[0,2]].values
x21 = iris_data.iloc[49:99:1,[0,2]].values
x31 = iris_data.iloc[99:len(iris_data):1,[0,2]].values
x12 = iris_data.iloc[0:49:1,[1,3]].values
x22 = iris_data.iloc[49:99:1,[1,3]].values
x32 = iris_data.iloc[99:len(iris_data):1,[1,3]].values

y = iris_data.iloc[0:len(iris_data):1,4].values

plt.figure()
plt.scatter(x11[0:50, 0], x11[0:50, 1], color="red", marker="o", label="setosa")
plt.scatter(x21[0:50, 0], x21[0:50, 1], color="blue", marker="x", label="versicolor")
plt.scatter(x31[0:50, 0], x31[0:50, 1], color="green", marker="^", label="virginica")
plt.title("sepal and petal length scatter")
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend(loc="upper left")

plt.figure()
plt.scatter(x12[0:50, 0], x12[0:50, 1], color="coral", marker="o", label="setosa")
plt.scatter(x22[0:50, 0], x22[0:50, 1], color="darkgreen", marker="x", label="versicolor")
plt.scatter(x32[0:50, 0], x32[0:50, 1], color="brown", marker="^", label="virginica")
plt.title("sepal and petal width scatter")
plt.xlabel("sepal width")
plt.ylabel("petal width")
plt.legend(loc="upper left")

plt.show()

