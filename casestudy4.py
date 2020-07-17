import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#1. We will use acoustic features to distinguish a male voice from female. Load the
#dataset from “voice.csv”, identify the target variable and do a one-hot encoding
#for the same. Split the dataset in train-test with 20% of the data kept aside for testing.
df=pd.read_csv("voice.csv")
#print(df.head())
#target variable
x=df.drop(["label"],axis=1)
y=df["label"]
from sklearn.preprocessing import LabelEncoder
u=LabelEncoder()
y=u.fit_transform(y)

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=2)

#2. Fit a logistic regression model and measure the accuracy on the test set.
from sklearn.linear_model import LogisticRegression
z=LogisticRegression()
z.fit(xtr,ytr)
ypre=z.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte,ypre))
#accuracy = 88%

#3. Compute the correlation matrix that describes the dependence between all
#predictors and identify the predictors that are highly correlated. Plot the
#correlation matrix using seaborn heatmap.
d=df.corr()
#print(d)
#sns.heatmap(d,annot=True)
plt.show()

#4. Based on correlation remove those predictors that are correlated and fit a
#logistic regression model again and compare the accuracy with that of
#previous model.
df1=df.drop(["meanfreq","sd","centroid","IQR","meandom"],axis=1)
x1=df1.drop(["label"],axis=1)

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x1,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression
s=LogisticRegression()
s.fit(x_tr,y_tr)
y_pre=s.predict(x_te)

from sklearn.metrics import classification_report
print(classification_report(y_te,y_pre))
