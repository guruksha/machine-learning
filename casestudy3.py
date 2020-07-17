import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
u=LinearRegression()
df=pd.read_csv("FyntraCustomerData.csv")
cr=df.corr()


#1. Compute -- Use seaborn to create a jointplot to compare the Time on Website
#and Yearly Amount Spent columns. Is there a correlation?
a=pd.DataFrame(df["Yearly_Amount_Spent"])
b=pd.DataFrame(df["Time_on_Website"])
sns.jointplot(a,b,kind="reg")
plt.show()
#yes there is a correlation


#2. Compute – Do the same as above but now with Time on App and Yearly
#Amount Spent. Is this correlation stronger than 1st One?
c=pd.DataFrame(df["Time_on_App"])
sns.jointplot(a,c,kind="reg")
plt.show()
#yes the correlation is stronger than time on website


#3. Compute -- Explore types of relationships across the entire data set using
#pairplot . Based off this plot what looks to be the most correlated feature with
#Yearly Amount Spent?
sns.pairplot(df)
plt.show()
#Length of membership is the most correlated feature

#4. Compute – Create linear model plot of Length of Membership and Yearly
#Amount Spent. Does the data fits well in linear plot?
sns.lmplot(x="Length_of_Membership",y="Yearly_Amount_Spent",data=df)
plt.show()
#Data fits well in linear plot

#5. Compute – Train and Test the data and answer multiple questions -- What is
#the use of random_state=85?
#Random_state=85 will generate a state which will have a specified set of training data
x=df[["Length_of_Membership"]]
y=df["Yearly_Amount_Spent"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=85)
u.fit(x_train,y_train)


#6. Compute – Predict the data and do a scatter plot. Check if actual and predicted data match?
y_predict=u.predict(x_test)
plt.scatter(y_predict, y_test)
plt.show()


#7. What is the value of Root Mean Squared Error?
err2=mean_squared_error(y_test,y_predict)
print(err2)

#8. Final Question –Based on coefficients interpret company should focus more
#on their mobile app or on their website.
#Company should focus on their mobile app
