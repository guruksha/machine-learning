#knn on breast cancer data setss
import pandas as pd
import seaborn as sns
df=pd.read_csv("breastcan.csv")
df.drop(["Unnamed: 32"],inplace=True,axis=1)
#print(df.head())
x1=df.drop(["diagnosis","id"],axis=1)
#print(x1.head())
y1=df["diagnosis"]
#print(y1.head())

from sklearn.preprocessing import StandardScaler,LabelEncoder
h=LabelEncoder()
g=StandardScaler()
x=g.fit_transform(x1)
df1=pd.DataFrame(x,columns=x1.columns)
y=h.fit_transform(y1)

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(df1,y,test_size=0.4,random_state=200)

from sklearn.neighbors import KNeighborsClassifier
u=KNeighborsClassifier()
u.fit(x_tr,y_tr)
ypre=u.predict(x_te)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_te,ypre))
