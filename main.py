import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datos=pd.read_csv('peliculas.csv')
df=pd.DataFrame(datos)

#Seleccionamos la columna a trabajar
x=df['Likes'].values
y=df['Ratings'].values

print("Valor promedio: ",df['Likes'].mean()) 
info=df[['Likes','Dislikes']].values
print(info)

X=np.array(list(zip(x,y)))
print(X)

kmeans=KMeans(n_clusters=5)
kmeans=kmeans.fit(X) 
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

colors=["m.","r.","c.","y.","b."]

for i in range(len(X)):
  print("Coordenada: ",X[i]," Label: ",labels[i])
  plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=5,zorder=10)
plt.show()