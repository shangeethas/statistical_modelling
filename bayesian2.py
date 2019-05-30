# libraries
import matplotlib.pyplot as plt
from scipy.io import loadmat

foetal_ecg = loadmat('foetal_ecg.mat')
remove_Channel1 = loadmat('removed_channel1.mat')
peak_fetus = loadmat('peaks_fetus.mat')

# load RealData_OptimumParams

data = foetal_ecg['2']
sampling_frequency = 250
time_interval = (data.size - 1) / sampling_frequency

baseline = LPfilter(data, 0.7 / sampling_frequency)
x1 = data - baseline
x = remove_Channel1
peaks = peak_fetus


def find(peaks):
    pass


I = find(peaks)
#[phase, phasepos] = PhaseCalculation(peaks)
#teta = 0
#pphase = PhaseShifting(phase, teta)

#dif_I = zeros(I.size - 1, 1)

mat_274 = loadmat('ecgca274_edfm.mat')
# _labels = mat_274.astype(np.uint16)
mat_748 = loadmat('ecgca748_edfm.mat')
mat_771 = loadmat('ecgca771_edfm.mat')
mat_997 = loadmat('ecgca997_edfm.mat')

print(mat_274)

plt.hist(mat_274, bins=100, density=True)
plt.xlabel('R Peak Value')
plt.ylabel('No of bins')
plt.title('Histogram of ECG Channel 274')
plt.savefig('274_ecg.png', dpi=128)
plt.close()
