import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D


datasets = pd.read_csv('datasets.csv', sep=';')
records = []
for i in range(0, 50):
    records.append([str(datasets.values[i,j]) for j in range(0, 14)])

df = DataFrame(records,columns=['state','population','infected','death','recovered','migration','politic','density','temperature','employment','poverly','netspeed','netcoverage','rurality'])
X = np.array(df[['population','infected','death','recovered','migration','politic','density','temperature','employment','poverly','netspeed','netcoverage','rurality']])
Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
