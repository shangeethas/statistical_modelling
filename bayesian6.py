import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import numpy as np
from scipy import signal
import math as mat

# reading as python dict
data_dict = loadmat('ecgca771_edfm.mat')

# extracting data array - the key is 'val'
data_array = data_dict['val']

# transpose for consistency
data_array = data_array.transpose(1, 0)

# convert to df
df = pd.DataFrame(data_array, columns=['ch' + str(n) for n in range(1, data_array.shape[1] + 1)])
# remove duplicates
# df = df.loc[~df.index.duplicated(keep='first')]
# same as
# ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6']

# visual inspection
df['ch3'].plot()
plt.title('Channel 3 Signal')
plt.savefig('out_signal3.png', dpi=128)
plt.close()

# Calculate relative amplitude dividing by 100
df_divided = df['ch3'] / 100
print(df_divided)

# calculate timing values
sampling_frequency = 1000
time_values = np.arange(0, df_divided.shape[0]) / sampling_frequency
print(time_values)


# function low pass filter
def low_pass_filter(input_data, fc):
    k = 0.7  # cut off value
    alpha = (1 - k * np.cos(2 * np.pi * fc) - np.sqrt(
        2 * k * (1 - np.cos(2 * np.pi * fc)) - k ** 2 * np.sin(2 * np.pi * fc) ** 2)) / (1 - k)
    y = signal.filtfilt(1 - alpha, [1, -alpha], input_data)
    return y


baseline = low_pass_filter(df_divided, 0.7 / sampling_frequency)
X = df_divided - baseline


# function peak detection
def peak_detection(input_data, heart_rate, fflag=0):
    N = input_data.shape[0]
    peaks = np.arange(0, input_data.shape[0])
    th = 0.5
    rng = mat.floor(th / heart_rate)

    if fflag:
        flag = fflag
    else:
        flag = abs(max(input_data)) > abs(min(input_data))

    if flag:
        for j in N:
            if (j > rng) and (j < N - rng):
                index = np.arange(j - rng, j + rng + 1)
            elif j > rng:
                index = np.arange(N - 2 * rng, N + 1)
            else:
                index = np.arange(1, 2 * rng + 1)

    if np.max(x[index]) == x[j]:
        peaks[j] = 1

    else:
        for j in N:
            if (j > rng) and (j < N - rng):
                index = np.arange(j - rng, j + rng + 1)
            elif j > rng:
                index = np.arange(N - 2 * rng, N + 1)
            else:
                index = np.arange(1, 2 * rng + 1)

            if min(x[index]) == x[j]:
                peaks[j] = 1

    # remove fake peaks
    I = find(peaks)
    d = diff(I)
    peaks[I[d < rng]] = 0

    return peaks


# Modeling maternal ECG
parameter_for_peak_detection = 1.35
# peaks1 = peak_detection(X, parameter_for_peak_detection/sampling_frequency)
# I = find(peaks1)

# visual inspection
X.plot()
plt.title('Maternal ECG Peaks')
plt.savefig('maternal_ecg_peaks.png', dpi=128)
plt.close()

t = np.linspace(0, 1.0, 2001)
low = np.sin(2 * np.pi * 5 * t)
high = np.sin(2 * np.pi * 250 * t)  
x = low + high

b, a = signal.butter(8, 0.125)
y = signal.filtfilt(b, a, x, padlen=150)
np.abs(y - xlow).max()
b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.
np.random.seed(123456)

n = 60
sig = np.random.randn(n) ** 3 + 3 * np.random.randn(n).cumsum()

gust = signal.filtfilt(b, a, sig, method="gust")
pad = signal.filtfilt(b, a, sig, padlen=50)
plt.plot(sig, 'k-', label='input')
plt.plot(gust, 'b-', linewidth=4, label='gust')
plt.plot(pad, 'c-', linewidth=1.5, label='pad')
plt.legend(loc='best')
plt.show()
