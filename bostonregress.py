import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
u=LinearRegression()
c=pd.read_csv("BostonHousing.csv")

################find correlation between attributes#############
#print(pd.isnull(c).sum())
#d=c.corr()
#a=sns.heatmap(d,annot=True)
# x=pd.DataFrame(c['medv'])
# y=pd.DataFrame(c[['rm','lstat','ptratio']])
#sns.distplot(c['medv'])
#sns.pairplot(c[['crim','zn','indus','chas']])

y=c['medv']
x=c[['rm','lstat','ptratio']]


#############find best fit of attributes###########
#sns.lmplot(x='medv',y='rm',data=c)
#sns.lmplot(x='medv',y='ptratio',data=c)
#sns.lmplot(x='medv',y='lstat',data=c)
#plt.show()

##########training and testing split#############
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=20)
u.fit(x_train,y_train)
#print(u.intercept_)
y_predict=u.predict(x_test)
#print(y_predict)

plt.scatter(y_predict,y_test)
scr=u.score(x_test,y_test)
#print(scr)
plt.show()
err=mean_absolute_error(y_test,y_predict)
err2=mean_squared_error(y_test,y_predict)
err3=r2_score(y_test,y_predict)
print(err)
print(err2)
print(err3)
