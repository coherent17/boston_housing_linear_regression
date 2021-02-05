import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.datasets import load_boston

def train_test_split(X,Y,test_size):
    X_train=X[:math.floor(len(X)*(1-test_size))]
    Y_train=Y[:math.floor(len(y)*(1-test_size))]
    X_test=X[math.floor(len(X)*(1-test_size)):]
    Y_test=Y[math.floor(len(Y)*(1-test_size)):]
    return X_train, X_test, Y_train, Y_test


boston_dataset = load_boston()
boston=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
boston['MEDV']=boston_dataset.target
#boston:
#        CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV
#0    0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98  24.0
#1    0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14  21.6
#2    0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03  34.7
#3    0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94  33.4
#4    0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33  36.2
#..       ...   ...    ...   ...    ...    ...   ...     ...  ...    ...      ...     ...    ...   ...
#501  0.06263   0.0  11.93   0.0  0.573  6.593  69.1  2.4786  1.0  273.0     21.0  391.99   9.67  22.4
#502  0.04527   0.0  11.93   0.0  0.573  6.120  76.7  2.2875  1.0  273.0     21.0  396.90   9.08  20.6
#503  0.06076   0.0  11.93   0.0  0.573  6.976  91.0  2.1675  1.0  273.0     21.0  396.90   5.64  23.9
#504  0.10959   0.0  11.93   0.0  0.573  6.794  89.3  2.3889  1.0  273.0     21.0  393.45   6.48  22.0
#505  0.04741   0.0  11.93   0.0  0.573  6.030  80.8  2.5050  1.0  273.0     21.0  396.90   7.88  11.9

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = boston['MEDV']
for i, col in enumerate(features):
    plt.subplot(1,2,i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
plt.show()

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

#building the model
w=np.zeros(2)
X_train_mean=np.mean(X_train)
Y_train_mean=np.mean(Y_train)
X_train_sqr_mean=np.mean(X_train**2)
X_train_Y_train_mean=np.mean(X_train*Y_train)

print(X_train_mean)
print(Y_train_mean)
print(X_train_sqr_mean)
print(X_train_Y_train_mean)


'''a=1
for t in range(0,len(X_train)):
    w[0]=w[0]-a*(w[0]+w[1]*X_train_mean-Y_train_mean)
    w[1]=w[1]-a*(w[0]*X_train_mean+w[1]*X_train_sqr_mean-X_train_Y_train_mean)
print(w[1])
print(w[0])'''

