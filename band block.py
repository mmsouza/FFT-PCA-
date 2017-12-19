from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


fs_Hz = 1000.0
bp_stop_Hz = np.array([20, 22])
b, a = signal.butter(2, bp_stop_Hz / (fs_Hz / 2.0), 'bandstop')

w, h = signal.freqz(b, a, 1000)
f = w * fs_Hz / (2 * np.pi)

base2 = pd.read_excel()  # load a dataset from excel file
temperature_24 = base2.loc[0:48, "NOx(GT)"]
B = temperature_24
# B=B.append(B)
# B=B.append(B)
# B=B.append(B)
# B=B.append(B)

plt.plot(np.arange(len(reject_outliers(temperature_24))), reject_outliers(temperature_24))

# plt.plot(np.arange(len(reject_outliers(B))), reject_outliers(B))


yf = signal.filtfilt(b, a, B)

print(np.correlate(yf, reject_outliers(temperature_24)))
# plt.plot(np.arange(48), yf[0:48])
plt.show()
