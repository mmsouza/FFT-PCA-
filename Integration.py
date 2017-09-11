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


def find_signature(name1, sig, stepdata, stepfft):
    # importing data
    data = pd.read_excel("AirQualityUCI.xlsx")
    print(len(data))
    # cleaning  null data
    # data = data.replace(-200.0, np.NAN)
    # data.dropna(inplace=True)
    # data.reset_index(drop=True, inplace=True)
    # signaturesList = []

    print(len(data.loc[:, "T"]))

    step2 = stepdata
    last, next = 0, step2
    k = 1
    name = name1
    list_list = []

    while last <= len(data.loc[:, name]):
        subset = data.loc[last:next, name]
        subset = reject_outliers(subset)

        if len(subset) > 0:
            N = len(subset)
            T = 1.0 / len(subset)
            yf = fft(detrend(subset))  # remove dc component to eliminate the freq zero
            xf = np.linspace(0, 1.0 / (2.0 * T), N // 2)
            yfft = N * np.abs(yf[0:N // 2])

            n = []
            m = name + ":"

            i = 1
            step = stepfft
            ii = step

            while i <= len(yfft):
                x = np.array(yfft[i:ii])
                if x.size != 0:
                    n.insert(len(n), (x.max(), str(i) + '-' + str(ii)))

                i, ii = ii + 1, ii + step
                n.sort()

            for z in n:
                m += z[1] + "#"
            if (m == sig):
                plt.figure(k)
                plt.plot(subset)
                k += 1
                plt.savefig(name + str(last) + str(next) + 'data.jpg')
                plt.close()
                #plt.figure(k)
                #plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
                #k += 1
                #plt.savefig(str(last) + str(next) + 'fft.jpg')

            #print("At:" + str(last) + "-" + str(next) + "Signatures:" + str(n))
            print(m)
            list_list.insert(len(list_list), m)
        last, next = next, next + step2

        # plt.show()

        # print(list_list)


if __name__ == "__main__":
    find_signature('C6H6(GT)' ,'C6H6(GT):81-100#61-80#41-60#21-40#1-20#',200,20)

    find_signature('PT08.S4(NO2)','PT08.S4(NO2):81-100#61-80#41-60#21-40#1-20#',200,20)

    find_signature('RH' ,'RH:81-100#61-80#41-60#21-40#1-20#',200,20)

    find_signature('PT08.S1(CO)', 'PT08.S1(CO):81-100#61-80#41-60#21-40#1-20#', 200, 20)

    find_signature('PT08.S2(NMHC)' ,'PT08.S2(NMHC):81-100#61-80#41-60#21-40#1-20#',200,20)

    find_signature('AH', 'AH:81-100#61-80#41-60#21-40#1-20#', 200, 20)


    find_signature('T' ,'T:81-100#61-80#41-60#21-40#1-20#',200,20)

    find_signature('PT08.S1(CO)', 'PT08.S1(CO):81-100#61-80#41-60#21-40#1-20#', 200, 20)

