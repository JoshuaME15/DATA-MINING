#------------------------------Joshua Hernández 1930693----------------------------------------#

from cProfile import label
from sklearn import linear_model
from ast import increment_lineno
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('train.csv', index_col=0)
#---area habitable---#
nombre = 'GrLivArea'
data = pd.concat([df['SalePrice'], df[nombre]], axis=1)
data.head(5)
data.plot.scatter(x=nombre, y='SalePrice', ylim=(0, 800000))

# plt.show()

#---venta y area de sotano---#
var = 'TotalBsmtSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

#---calidad de la construccion (Categorica)---#
var = 'OverallQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.head()
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
# plt.show()

#---año de construcion y calidad---#
var = 'YearBuilt'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.head()
f, ax = plt.subplots(figsize=(20, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)


#---Matriz de coorelacion & Matriz con todas las coorelaciones---#
coorrelacion_de_matriz = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(coorrelacion_de_matriz, vmax=.8, square=True)
# plt.show()

#---RL por modelo y prediccion---#
regresion = linear_model.LinearRegression()
area = df['GrLivArea'].values.reshape((-1, 1))

modelo = regresion.fit(area, df['SalePrice'])


print("Interseccion (b) ", modelo.intercept_)
print("Pendiente (m)", modelo.coef_)

entrada = [[500], [900], [1200], [1600], [2100], [
    2600], [3000], [3500], [3600], [4500], [5000], [400]]

modelo.predict(entrada)

plt.scatter(entrada, modelo.predict(entrada), color="red")
plt.plot(entrada, modelo.predict(entrada), color="black")

plt.ylabel('Precio de venta')
plt.xlabel(" habitable")
plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.show()


#----RL por pasos----#

x = df['GrLivArea']
y = df['SalePrice']

n = len(x)
x = np.array(x)
y = np.array(y)

sumx = sum(x)

sumy = sum(y)

sumx2 = sum(x*x)

sumy2 = sum(y*y)

sumxy = sum(x*y)

promx = sumx/n

promy = sumy/n

m = (sumx*sumy - n * sumxy)/(sumx**2 - n * sumx2)

b = promy - m * promx
sigmax = np.sqrt(sumx2/n - promx**2)
sigmay = np.sqrt(sumy2/n - promy**2)
sigmaxy = sumxy/n - promx * promy
R2 = (sigmaxy/(sigmax*sigmay))**2

print(R2)
print(m)

plt.plot(x, y, 'o', label='Datos')
plt.plot(x, m * x + b, label='Ajuste')
plt.xlabel('Area en m2 a habitable')
plt.ylabel('Precio de venta')
plt.grid()
plt.legend()
plt.show()