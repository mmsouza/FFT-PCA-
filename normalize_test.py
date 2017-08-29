import numpy.fft as np_fft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.fftpack import fft
from scipy.signal import detrend
import matplotlib.dates as dates
from sklearn.preprocessing import normalize


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


base2 = pd.read_excel("AirQualityUCI.xlsx")

temperature_24 = base2.loc[0:48, "T"]

norm_temp = pd.DataFrame()
print(np.max(temperature_24))

i=0

col_init = 3
col_final = 5

subset = base2.iloc[0:300, col_init:col_final]

subset = subset.replace(-200.0, np.NAN)
subset.dropna(inplace=True)
subset.reset_index(drop=True,inplace=True)



f, array = plt.subplots(col_final-col_init, sharex=True, sharey=True)

#ax1.set_title('Sharing both axes')
#ax2.scatter(x, y)
#ax3.scatter(x, 2 * y ** 2 - 1, color='r')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.


for column in subset:
    B = subset[column]


   # B = reject_outliers(B)
    array[i].set_title(B.name,fontsize= 8)
    array[i].plot(B.apply(lambda x: x/B.max()))
    array[i].grid()
    i = i+1

f.canvas.set_window_title('Normalized data')
f.subplots_adjust(hspace=1)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)


f2, array2 = plt.subplots(col_final-col_init, sharex=True, sharey= False)
i=0
for column in subset:
    B = subset[column]
    #if(B[column] == -200):
    print(type(B))
    #B = reject_outliers(B)
    array2[i].set_title(B.name,fontsize= 8)
    array2[i].plot(B)
    i = i+1

f2.canvas.set_window_title('original data')
f2.subplots_adjust(hspace=1)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

'''''
t= temperature_24.apply(lambda x: x/temperature_24.max() )
#for var in temperature_24:
   # norm_temp.append(var/temperature_24.max(), ignore_index=True)
   #print( pd.DataFrame(var) )

plt.plot(t)
plt.figure(2)
plt.plot(temperature_24)
plt.show()


s=300

plt.figure(3)
plt.plot(base2.loc[0:s,"NO2(GT)"])

plt.figure(4)
plt.plot(reject_outliers(base2.loc[0:s,"NO2(GT)"]))
'''

plt.show()