import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.io import loadmat

mat_274 = loadmat('ecgca274_edfm.mat')
print(type(mat_274))
print(mat_274)
quit()

plt.hist(mat_274, bins=1, density=True)
plt.xlabel('ECG Value')
plt.ylabel('No of bins')
plt.title('ECG Value of Channel 274')
plt.savefig('274_ecg.png', dpi=128)
plt.close()
