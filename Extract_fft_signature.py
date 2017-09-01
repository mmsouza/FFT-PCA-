import numpy.fft as np_fft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.fftpack import fft
from scipy.signal import detrend
import matplotlib.dates as dates

data = pd.read_excel("AirQualityUCI.xlsx")

# cleaning the null data
data = data.replace(-200.0, np.NAN)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# selecting a column to do the fft transformation
temperature_24 = data.loc[:, "T"]

# performing fft
# Number of sample points
N = len(temperature_24)
# sample spacing
T = 1.0 / len(temperature_24)
yf = fft(detrend(temperature_24))  # remove dc component to eliminate the freq zero
xf = np.linspace(0, 1.0 / (2.0 * T), N // 2)

# plot space domain
plt.figure(1)
plt.plot(temperature_24)
# plot frequency domain
plt.figure(2)
plt.plot(xf, N * np.abs(yf[0:N // 2]))




yfft = N * np.abs(yf[0:N // 2])

n = []
i = 1
step = 20
ii = step

while i <= len(yfft):
    x = np.array(yfft[i:ii])
    n.insert(len(n),(x.max(), str(i)+'-'+str(ii)))
    i, ii = ii+1, ii+step
n.sort()
print(n)

plt.show()