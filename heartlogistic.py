import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("heart.csv")
#countm=len(df[df["target"]== 1 ])
#perfml=df[df["sex"]==1].count()
#permale=len(df[df["sex"]== 0])

#sns.countplot(df["age"],data=df,hue="target")
#sns.countplot(df["age"],data=df,hue="sex")
d=df.corr()
#sns.heatmap(d,annot=True)
#pd.crosstab(df["age"],df["target"]).plot(kind="bar")
#sns.heatmap(df.isnull(),yticklabels=False)
plt.show()

#sns.countplot(df["sex"],data=df,hue="target")
plt.show()

CP=pd.get_dummies(df["cp"],prefix="cp")
SLO=pd.get_dummies(df["slope"],prefix="sl")
CA=pd.get_dummies(df["ca"],prefix="CA")
THAL=pd.get_dummies(df["thal"],prefix="THAL")
df=pd.concat([df,CP,SLO,CA,THAL],axis=1)
df.drop(["cp","ca","thal","slope"],axis=1,inplace=True)
#print(df.columns)

x=df[["age","sex","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","cp_0","cp_1","cp_2","cp_3","sl_0","sl_1","sl_2","CA_0","CA_1","CA_2","CA_3","CA_4","THAL_0","THAL_1","THAL_2","THAL_3"]]
y=df[["target"]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)

from sklearn.linear_model import LogisticRegression
u=LogisticRegression()
u.fit(x_train,y_train)
y_predict=u.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_predict))

