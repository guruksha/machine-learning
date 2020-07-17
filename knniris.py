import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Iris.csv")
df.drop(["Unnamed: 0"],inplace=True,axis=1)
#sns.pairplot(df,hue="Species")
plt.show()
x=df[["Id","SepalLengthCm","PetalLengthCm","PetalWidthCm"]]
y=df["Species"]
#print(y)

from sklearn.preprocessing import LabelEncoder
u=LabelEncoder()
y=u.fit_transform(y)
#print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
l=[]
f=[]
for i in range(1,100,2):
    obj=KNeighborsClassifier(n_neighbors=i)
    obj.fit(x_train,y_train)
    y_predict=obj.predict(x_test)
    acc=accuracy_score(y_test,y_predict)
    l.append(acc)
    f.append(i)
#print(confusion_matrix(y_test,y_predict))
sns.lineplot(x=f,y=l)
plt.show()
