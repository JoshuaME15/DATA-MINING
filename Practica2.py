#------------------------------Joshua Hernández 1930693----------------------------------------#

from ast import Lambda
from gettext import npgettext
from posixpath import split
import re
from tkinter import Scale
from types import LambdaType
import numpy as np
import datetime
import pandas as pd
import time as i

df = pd.read_csv('mod7.csv', index_col=False)


df['Release Date'] = df['Release Date'].str.replace("  ", " ")
df['Release Date'] = df['Release Date'].str.replace(" ", "/")

df['Release Date'] = pd.to_datetime(df['Release Date'])

df['Release Date'] = pd.to_datetime(
    df['Release Date'], format='%m/%d/%Y %-H:%M:%S')


#df['Movie Runtime'] = df['Movie Runtime'].str.replace("hr", ":")
#df['Movie Runtime'] = df['Movie Runtime'].str.replace("min", ":")
df['Movie Runtime'] = df['Movie Runtime'].str.replace(",", ":")
df['Movie Runtime'] = df['Movie Runtime'].str.replace(" ", "")


def add(x):
    return ("0" + x + ":00")


df['Movie Runtime'] = df['Movie Runtime'].apply(add)



"""-----------------------------------------------------------------
 Separacion de la duracion de la pelicula por horas y minutos
df[['Hrs', 'Min']] = df['Movie Runtime'].str.split(",", expand=True)
df['Movie Runtime'] = df['Movie Runtime'].astype("datetime64")
t = datetime.time(df['Hrs'], df['Min'])
df['ti'] = df['ti'].t

from_data = "%d/%m/%y %H:%M:%S
-------------------------------------------------------------------"""
