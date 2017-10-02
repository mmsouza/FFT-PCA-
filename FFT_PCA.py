import fft
from sklearn import decomposition
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.axes import  Axes

#data = pd.read_excel("AirQualityUCI.xlsx")
data = pd.read_csv("normal_24.csv")

print(len(data.columns))

step = 200
last, next = 0, step
list_dataframes = []
list_aux = []
col = 0

while col <= len(data.columns) -1:
    last, next = 0, step
    while last <= len(data):
        subset = data.iloc[last:next, col]
        if len(subset) > 0:
            xf, yff, yf = fft.hf_fft(subset)
            yff = yff.tolist()
            yff.insert(0, subset.name)
            list_aux.insert(len(list_aux), yff)
        last, next = next, next + step
    list_dataframes.insert(len(list_dataframes), pd.DataFrame(list_aux))
    list_aux = []
    col += 1



print(len(list_dataframes))

list_aux = []
for i in range(0 , 41):
    dfaux= (list_dataframes[i].loc[:, 1:199])
    dfaux.dropna(inplace=True)
    dfaux = pd.DataFrame(normalize(dfaux))

    list_aux.insert(len(list_aux),dfaux)

dftotal = pd.concat(list_aux,ignore_index= True)


#writer = pd.ExcelWriter('pandas_simple2.xlsx', engine='xlsxwriter')
#dftotal.to_excel(writer, sheet_name='Sheet1')
#writer.save()




pca = decomposition.PCA(n_components = 2)
pca.fit(dftotal)


for i in range(0 , 41):
    dfaux=list_dataframes[i].loc[:, 1:199]
    dfaux.dropna(inplace= True)
    dfaux = normalize(dfaux)

#print(type(dfaux))
    X = pca.transform(dfaux)
#print(pca.components_)

    plt.figure(i)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.scatter(X[:, 0], X[:, 1])

plt.show()
