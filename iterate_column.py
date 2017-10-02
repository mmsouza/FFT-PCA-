import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import OnehotTransactions
import numpy as np
from scipy.signal import detrend
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


# importing data
data = pd.read_excel("AirQualityUCI.xlsx")
print(len(data))
# cleaning  null data
# data = data.replace(-200.0, np.NAN)
# data.dropna(inplace=True)
# data.reset_index(drop=True, inplace=True)
# signaturesList = []


step2 = 400
step = 10
last, next = 0, step2

transactions_list = []
k = 1
while last <= len(data):

    col = 2  # define initial column
    signature_list = []
    while col <= len(data.columns) - 1:
        # print(col)
        subset = data.iloc[last:next, col]
        #subset = reject_outliers(subset)

        if len(subset) > 0:
            plt.figure(k)
            plt.plot(subset)
            k += 1
            plt.savefig(str(last) + str(next) + 'data.jpg')

            N = len(subset)
            T = 1.0 / len(subset)
            yf = fft(detrend(subset))  # remove dc component to eliminate the freq zero
            xf = np.linspace(0, 1.0 / (2.0 * T), N // 2)
            yfft = N * np.abs(yf[0:N // 2])

            plt.figure(k)
            plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
            k += 1
            plt.savefig(str(last) + str(next) + 'fft.jpg')

            n = []
            m = subset.name + ":"

            i = 1

            ii = step

            while i <= len(yfft):
                x = np.array(yfft[i:ii])
                if x.size != 0:
                    n.insert(len(n), (x.max(), str(i) + '-' + str(ii)))

                i, ii = ii + 1, ii + step
                #n.sort()

            for z in n:
                m += z[1] + "#"

                # print("name:" + subset.name + " At:" + str(last) + "-" + str(next) + "Signatures:" + str(n))
            print(m)
            signature_list.insert(len(signature_list), m)
        col += 1
    transactions_list.insert(len(transactions_list), signature_list)
    last, next = next, next + step2

# for l in transactions_list:
#    for sig in l:
#        print(sig)
#    print("\n")

'''
oht = OnehotTransactions()
oht_ary = oht.fit(transactions_list).transform(transactions_list)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

pd.set_option('display.max_columns', None)
with pd.option_context('display.max_rows', None, 'display.max_columns', None ):
    print(frequent_itemsets)

print(association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9))
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9).to_excel('path_to_file.xlsx', sheet_name='Sheet1')



'''
