

####project on knn iris

#import necessary packages
import pandas
import numpy
import matplotlib.pyplot as plt

#read data set
data=pandas.read_csv(r'D:\iris.csv')

#data understanding

#print(data)
#print(data.head())
###print(data.tail())
#print(data.columns)
##print(data.shape)
##print(data['sl'])
###print(data['class'])
###print(data['pw'])
###print(data)
##
#data visulations
##x=data['sl']
##y=data['sw']
##plt.scatter(x,y)
##plt.xlabel('sepel length')
##plt.ylabel('sepel width')
##plt.title('sepel length vs sepel width')
##plt.show()
##
##x=data['sl']
##y=data['pl']
##plt.scatter(x,y)
##plt.xlabel('sepel length')
##plt.ylabel('petal length')
##plt.title('sepel length vs petal lengt')
##plt.show()
##
###data preprocessing
##
###data=data.fillna(1)
###data=data.dropna()
##
x=data.iloc[:,:-1].values#independent variables
y=data.iloc[:,-1].values#target variable
###print(y)
###print(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=9)
###print(xtrain.shape)
##
###model building
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
##
###accuracy checking
y_pred=model.predict(x_test)
###print(y_pred)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)

###result\future prediction
print(model.predict([[5.1,3.5,1.4,0.2]]))
##
##
##
