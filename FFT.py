import  numpy.fft as np_fft
import numpy as np
import  pandas as  pd
import  matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.fftpack import fft





def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


base = pd.read_csv('AirQualityUCI.csv')
base2 = pd.read_excel("AirQualityUCI.xlsx")
filtred=reject_outliers(pd.DataFrame(base2, columns={'T'}))
# Number of sample points
N = 9357
# sample spacing
T = 1.0 /9357.0

plt.figure(2)
plt.plot(pd.DataFrame(base2, columns={'Date'}), filtred)


filtred2=reject_outliers(pd.DataFrame(base2, columns={'NOx(GT)'}))



plt.figure(1)
yf = fft(filtred)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()


plt.figure(1)
yf = fft(filtred2)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()

plt.show()




#plt.scatter(np.arange(len(base2['T'])), base2['T'])




plt.show()

print(len(base2['Date']))
print(len(base2['T']))






'''''
plt.figure(1)

plt.plot(pd.DataFrame(base2, columns={'Date'})  ,pd.DataFrame(base2, columns={'T'}))
filtred=reject_outliers(pd.DataFrame(base2, columns={'T'}))
#filred=sps.medfilt(pd.DataFrame(base2, columns={'T'}),kernel_size= 7)
plt.figure(2)
plt.plot(pd.DataFrame(base2, columns={'Date'}),filtred)
plt.show()


#print(base.head())
#print(pd.DataFrame(base, columns={'CO(GT)'}))

#plt.plot(pd.DataFrame(base, columns={'Date'}))

#plt.show()
'''
