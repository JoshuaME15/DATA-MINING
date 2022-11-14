#------------------------------Joshua Hernández 1930693----------------------------------------#

from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import line
from sklearn.cluster import KMeans


df = pd.read_csv('train.csv', index_col=0)
x = df['SalePrice'].values
y = df['OverallQual'].values
X = np.array(list(zip(x, y)))

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
color = ['m.', 'r.', 'c.', 'y.', 'b.']

for i in range(len(X)):
    print('Coordenada: ', X[i], 'Label: ', labels[i])
    plt.plot(X[i][0], X[i][1], color[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='+', s=150, linewidths=1, zorder=10)
plt.show()