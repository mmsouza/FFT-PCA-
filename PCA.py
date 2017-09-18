from sklearn import decomposition
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

df = pd.read_excel("AirQualityUCI.xlsx")
df2 = df.iloc[:, 3:11]
df2= normalize(df2)
pca = decomposition.PCA(n_components= 2)
X = pca.fit_transform(df2)
print(pca.components_)

plt.scatter(X[:,0], X[:,1])
plt.show()

'''
plt.plot(pca[:,0],pca[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(pca[20:40,0], pca[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples via sklearn.decomposition.PCA')
plt.show()
'''