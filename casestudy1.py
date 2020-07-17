import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
u=LinearRegression()

#1)Load the data from “cereal.csv” and plot histograms of sugar and vitamin content across different cereals.
a=pd.read_csv("cereal.csv")
b=a['sugars']
plt.hist(b)
plt.show()
c=a["vitamins"]
plt.hist(c)
plt.hist(c)
plt.show()
#2)The names of the manufactures are coded using alphabets,
# create a new column with their full name using the below mapping.#
dic={'N': 'Nabisco',
'Q': 'Quaker Oats',
'K': 'Kelloggs',
'R': 'Raslston Purina',
'G': 'General Mills' ,
'P' :'Post' ,
'A':'American Home Foods Products'
}
l=[]
for q in a['mfr']:
    l.append(dic[q])
a['fn']=l

#Create a bar plot where each manufacturer is on the y axis and #
# the height of the bars depict the number of cereals manufactured by them. #
ax=sns.countplot(x=a['fn'])
plt.show()

#3. Extract the rating as your target variable ‘y’ and all numerical parameters as your predictors ‘x’.
# Separate 25% of your data as test set.
x=a[['calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups']]
y=a['rating']
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.25,random_state=25)

#4)Fit a linear regression module and measure the mean squared error on test dataset.
u.fit(x_tr,y_tr)
y_pre=u.predict(x_te)
err=mean_squared_error(y_te,y_pre)
print(err)
