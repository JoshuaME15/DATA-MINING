#------------------------------Joshua Hernández 1930693----------------------------------------#

from ast import Lambda
from itertools import count
from turtle import color
from matplotlib import colorbar, colors
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('titanic2.csv', index_col=0)
d = {'male': 'M', 'female': 'F'}
df['Sex'] = df['Sex'].apply(lambda x: d[x])

des = pd.crosstab(df.Survived, df.Sex)


pclass_genero_survcount = df.groupby(['Pclass', 'Sex'])['Survived'].sum()

 fig = plt.figure(figsize=(10, 8))  # Figura con tamaño

# --------------------------------------Cuentas y porcentaje--------------------#
# Para ver una grafica a lado de otra
plt.subplot2grid((1, 3), (0, 0))
df['Survived'].value_counts().plot(kind='bar', alpha=0.5)
plt.title('Sobrevivieron - Cuenta total')


plt.subplot2grid((1, 3), (0, 2))
df['Survived'].value_counts(normalize=True).plot(kind='bar', alpha=0.5)
plt.title('Sobrevivieron - Porcentaje total')

# ---------------------Sobrevivieron mas Hombres o Mujeres?---------------------#
fig = plt.figure(figsize=(10, 8))
df['Sex'][df['Survived'] == 1].value_counts(
 normalize=True).plot(kind='barh', alpha=0.5)
plt.title('Sobrevivieron - M-F - ')


"""---------------------------------------------------------------
Influyo?

fig = plt.figure(figsize=(10, 5))
df['Pclass'][df['Survived'] == 1].value_counts(
normalize=True).plot(kind='bar', alpha=0.5)
plt.title('Sobrevivientes por Clase ')

----------------------------------------------------------------"""

# ---------------------------------------Edad Economia----------------------------------------#
fig = plt.figure(figsize=(20, 10))

for t_class in [1, 2, 3]:
    df['Age'][df['Pclass'] == t_class].plot(kind="kde")

plt.legend(('1ra.Clase', '2da.Clase', '3ra.Clase'))

plt.show()


name_survive = df.loc[df['Survived'] == 1].count()
name_dead = df.loc[df['Survived'] == 0].count()


#df.set_index(['Pclass'], inplace=True)
# df.sort_index(inplace=True)
p = df.groupby(by=['Pclass'])
counts = df['Survived'].value_counts()
tot_y = df['Survived'].value_counts().sum()
#p = df.groupby(by=['Sex']).count().sum()

classs = df.sort_values(by=['Pclass'])