#logistic regression on breast cancer dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("breastcan.csv")
#print(df.head())
#d=df.corr()
#sns.heatmap(d,annot=True)
#sns.heatmap(df.isnull(),yticklabels=False)
plt.show()
#print(df.columns)
Dia=pd.get_dummies(df["diagnosis"])
Dia.drop(["B"],inplace=True,axis=1)

df1=pd.concat([df,Dia],axis=1)
df1.drop(["id","diagnosis","Unnamed: 32"],axis=1,inplace=True)
#print(df1.head())
x=df1.drop(["M"],axis=1)
y=df1["M"]
#print(y)

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.3,random_state=23)

from sklearn.linear_model import LogisticRegression
u=LogisticRegression()
u.fit(x_tr,y_tr)
ypre=u.predict(x_te)

from sklearn.metrics import classification_report
print(classification_report(y_te,ypre))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_te,ypre))
