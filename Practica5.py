#------------------------------Joshua Hernández 1930693----------------------------------------#

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv', index_col=0)


media = round(df['SalePrice'].mean(), 1)
desviacion_estandar = round(np.std(df['SalePrice']), 1)
Tamano_Muestra = len(df)

print(media)  # 180921.2
print(desviacion_estandar)  # 79415.3
print(Tamano_Muestra)  # 1460

""" Prueba de hipotesis
    Hipotesis
    Suponga que 1460 casa cuestan en promedio  180921.019 con una desviacion estandar de 79442
    Probar la hipotesis nula  M = 200000 contra la hipotesis alternativa de M < 200000 """


a = df['SalePrice']

mean, std = 200000, 79415.3

# Calcular la media muestral
sample_mean = a.mean()

# Calcular el error estandar
se = std / np.sqrt(len(a))
# Estadistica de Z
Z = (sample_mean-mean)/se

print("La estradistica de Z es {}".format(Z))

# Valor de P
P = 2*stats.norm.sf(abs(Z))
""" Aqui stats.nom.sf devuelve el area en el lado derecho de Z
 Por lo que 2* es la suma de las areas en los lados izquierdo y derecho de Z """
print("El valo de p es {}".format(P))

print("Se rechaza la hipotesis nula ya que   -9.1795 < -1.96 ")
ax = sns.distplot(df['SalePrice'])
plt.show()