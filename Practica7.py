#------------------------------Joshua Hernández 1930693----------------------------------------#

from cProfile import label
from joblib import PrintTime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv', index_col=0)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['YrSold'], df['SalePrice'])
plt.xlabel("Perdiodos", fontsize=14)
plt.ylabel("Ventas", fontsize=14)

plt.title("Serie de tiempo", fontsize=18)

# Datos anuales

nombre = 'YearBuilt'
data = pd.concat([df['SalePrice'], df[nombre]], axis=1)

anual = data.groupby(by=['YearBuilt']).sum().reset_index()


plt.figure(figsize=(12, 6))
plt.plot(anual.index, anual['SalePrice'], '-o')

plt.xlabel("Perdiodos", fontsize=14)
plt.ylabel("Ventas", fontsize=14)

plt.title("Serie de tiempo", fontsize=18)


# Media Movil
anual["PR"] = anual['SalePrice'].rolling(window=3).mean().shift(1)


plt.figure(figsize=(12, 6))
plt.plot(anual.index, anual['SalePrice'], '-o', color='black', label='data')
plt.plot(anual.index, anual['PR'], '-o', color='red', label='Pronostico')
plt.xlabel("Perdiodos", fontsize=14)
plt.ylabel("Ventas", fontsize=14)
plt.legend(loc='best')

plt.title("Serie de tiempo sin prediccion", fontsize=18)
plt.show()

# Agregar una nuieva fila
anual.loc[len(anual)] = [int(anual.iloc[len(anual)-1][0])+1, 0, 0]
anual["PR"] = anual['SalePrice'].rolling(window=3).mean().shift(1)
anual['YearBuilt'] = anual['YearBuilt'].astype(int)
anual['SalePrice'] = anual['SalePrice'].astype(float)
anual['PR'] = anual['PR'].astype(float)

plt.figure(figsize=(12, 6))
plt.plot(anual['YearBuilt'][:-1], anual['SalePrice']
         [:-1], '-o', color='black', label='data')
plt.plot(anual['YearBuilt'], anual['PR'],
         '-o', color='red', label='Pronostico')
plt.xlabel("Años", fontsize=14)
plt.ylabel("Ventas", fontsize=14)
plt.legend(loc='best')

plt.title("Prediccion", fontsize=18)
plt.show()