import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
u=LinearRegression()
c=pd.read_csv("gdp_per_capita.csv",encoding='cp1252')
d=pd.read_csv("oecd_bli_2015.csv")
a=c.merge(d,on='Country')
a.dropna(axis=1,how='all',inplace=True)
a.dropna(axis=0,how='any',inplace=True)

##################correlation####################
d=a.corr()
#sns.heatmap(d,annot=True)
plt.show()


##############model###############
x=pd.DataFrame(a['2015'])
y=pd.DataFrame(a['Value'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)
u.fit(x_train,y_train)
y_predict=u.predict(x_test)
scr=u.score(x_test,y_test)
print(scr)
err=mean_absolute_error(y_test,y_predict)
err2=mean_squared_error(y_test,y_predict)
err3=r2_score(y_test,y_predict)
print(err)
print(err2)
print(err3)


