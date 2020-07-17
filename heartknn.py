import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("heart.csv")
from sklearn.preprocessing import StandardScaler
g=StandardScaler()
x1=df[["age","sex","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","cp","slope","ca","thal"]]
y=df["target"]
x=g.fit_transform(x1)
df1=pd.DataFrame(x,columns=x1.columns)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df1,y,test_size=0.3,random_state=12)

from sklearn.neighbors import KNeighborsClassifier
u=KNeighborsClassifier(n_neighbors=1)
u.fit(x_train,y_train)
y_pre=u.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pre))
