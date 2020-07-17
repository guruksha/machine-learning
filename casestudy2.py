import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("prisoners.csv")

#1. Data Loading:
#a. Load the dataset “prisoners.csv” using pandas and display the first and
#last five rows in the dataset.

print(df.head())
print(df.tail())

#b. Use describe method in pandas and find out the number of columns. Can
#you say something about those rows who have zero inmates?
print(df.describe())
print(df.loc[(df["No. of Inmates benefitted by Elementary Education"]== 0) &
             (df["No. of Inmates benefitted by Adult Education"] == 0) &
             (df["No. of Inmates benefitted by Higher Education"]==0) &
             (df["No. of Inmates benefitted by Computer Course"]==0)])


#2. Data Manipulation:
#a. Create a new column -’total_benefitted’ that is a sum of inmates benefitted through all modes.
df['total benefitted']=df.iloc[:,-4:].sum(axis=1)
print(df.head())

#b. Create a new row - “totals” that is the sum of all inmates benefitted through each mode across all states.
df.loc["total"]=df.sum(numeric_only=True)


#3. Plotting:
#a. Make a bar plot with each state name on the x -axis and their total
#benefitted inmates as their bar heights. Which state has the maximum number of beneficiaries?
df.dropna(inplace=True)
plt.bar(df['STATE/UT'],df['total benefitted'])
plt.show()

#b. Make a pie chart that depicts the ratio among different modes of
#benefits.
df.loc['total']=df.sum(numeric_only=True)
plt.pie(df.loc["total"][3:],labels=["EE","AE","HE","CC"])
plt.show()
