#------------------------------Joshua Hernández 1930693----------------------------------------#

from numpy import int64
import pandas as pd


df = pd.read_csv('titanic2.csv')

nombres = df['Name']
nombre_ = df.Name

df = df[[
    'Name',
    'Sex',
    'Age',
    'Siblings/Spouses Aboard',
    'Parents/Children Aboard',
    'Fare',
    'Survived',
    'Pclass']]

df['Age'] = df['Age'].astype(int64)

count = df['Age'].value_counts()
survive = df['Survived'].value_counts()

df['SurviveLabel'] = df['Survived'].map({0: 'Murio', 1: 'Sobrevivio'})
tot = df['SurviveLabel'].value_counts()


porcentaje_supervivencia = (df['Survived'].mean(skipna=True))*(100)

porcentaje_supervivencia_porclase = df.groupby('Pclass').mean()

class_sex = df.groupby(['Pclass', 'Sex']).mean()


print(porcentaje_supervivencia)
print(porcentaje_supervivencia_porclase)

print(class_sex)