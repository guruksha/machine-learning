import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df=pd.read_csv("loantrain.csv")
#print(df.isnull().sum())
#sns.heatmap(df.isnull(),yticklabels=False)
#sns.countplot(df["Dependents"],data=df)
#sns.countplot(df["Gender"],data=df)
#sns.countplot(df["Married"],data=df)
#sns.countplot(df["Property_Area"],data=df,hue="Loan_Status")
#sns.countplot(df["Self_Employed"],data=df)
#sns.countplot(df["CoapplicantIncome"],data=df,hue='Loan_Status')
#sns.distplot(df["ApplicantIncome"].dropna())
plt.show()
def add_loanamt(cols):
     LoanAmount=cols[0]
     if pd.isnull(LoanAmount):
         return int(df["LoanAmount"].mean())
     else:
         return LoanAmount
df["LoanAmount"] = df[["LoanAmount"]].apply(add_loanamt,axis=1)


def add_loanamterm(cols):
     LoanAmounterm=cols[0]
     if pd.isnull(LoanAmounterm):
         return int(df["Loan_Amount_Term"].mean())
     else:
         return LoanAmounterm
df["Loan_Amount_Term"] = df[["Loan_Amount_Term"]].apply(add_loanamterm,axis=1)
gender=pd.get_dummies(df["Gender"],prefix="G",drop_first=True)
depe=pd.get_dummies(df["Dependents"],prefix="dep")
slfemp=pd.get_dummies(df["Self_Employed"],prefix="SE",drop_first=True)
proparea=pd.get_dummies(df["Property_Area"])
target=pd.get_dummies(df["Loan_Status"],drop_first=True)
ed=pd.get_dummies(df["Education"],drop_first=True)

df = pd.concat([df,gender,depe,slfemp,proparea,target,ed],axis=1)
df.dropna(inplace=True)
df.drop(["Loan_ID","Education","Married","Self_Employed","Gender","Dependents","Loan_Status","Self_Employed","Property_Area"],axis=1,inplace=True)
print(df.columns)
x=df[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","G_Male","dep_0","dep_1","dep_2","dep_3+","SE_Yes","Credit_History","Rural","Semiurban","Urban"]]
y=df[["Y"]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)

from sklearn.linear_model import LogisticRegression
u=LogisticRegression()
u.fit(x_train,y_train)
y_pr=u.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pr))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pr))
