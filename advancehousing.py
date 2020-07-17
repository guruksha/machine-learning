#cleaning of data removing all the null values except FirePlaceQu
import pandas as pd
df=pd.read_csv("C://ml//house-prices-advanced-regression-techniques//train.csv")
#print(df.columns)
#removing columns having most of the nan values
pd.set_option('display.max_rows', None)
#print(a.head())
b=df['LotFrontage'].mean()
df['LotFrontage']=df['LotFrontage'].fillna(b)
#print(a)
df=df.dropna(subset=['BsmtQual','BsmtExposure','BsmtFinType2','Electrical','MasVnrType','MasVnrArea','GarageType'])
df.drop(['PoolQC','MiscFeature','PoolArea'],axis=1,inplace=True)
#print(a.head())
df['Alley']=df['Alley'].fillna("NOTA")
df['Fence']=df['Fence'].fillna("NOTA")
df['Fenc']=df['Fence'].fillna("NOTA")
print(pd.isnull(df).sum())

