#import library yang akan digunakan
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as numpy
import pandas as pd
from sklearn.cluster import  KMeans

#menyiapkan data dan memanggil dataset
dataset = pd.read_csv('E:\konsumen.csv')
dataset.keys()
dataku = pd.DataFrame(dataset)
dataku.head()

#konversi ke data array
X = np.asarray(dataset)
print(X)

#menampilkan data ke dalam bentuk scatter plot
plt.scatter(X[:,0], [:,1], label='True Position')
plt.xlabel("Gaji")
plt.ylabel("Pengeluaran")
plt.title("Grafik Penyebaran Data Konsumen")
plt.show()

#mengaktifkan K-Means dengan jumlah K=2
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

#menampilkan nilai centroid yang digenerate oleh algoritma
print(kmeans.cluster_centers_)

#plot data point
#memvisualisasikan hasil klasterisasi data konsumen
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.xlabel("Gaji")
plt.ylabel("Pengeluaran")
plt.title("Grafik Hasil Klasterisasi Data Gaji dan Pengeluaran Konsumen")
plt.show()

#plot data point
#memvisualisasikan hasil klasterisasi dengan centroid dr masing2 kluster
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')
plt.xlabel("Gaji")
plt.ylabel("Pengeluaran")
plt.title("Grafik Hasil Klasterisasi Data Gaji dan Pengeluaran Konsumen")
plt.show()