#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat
from matplotlib.pyplot import figure

list_files = ['SPECT.train', 'SPECT.test']
for file in list_files:
    df = pd.read_csv(file)
    print(file, 'size:', df.shape)

headers1 = ['overall_diagnosis']
headers2 = ['F'+ str(n) for n in range(1, 23)]
headers1.extend(headers2)
print(headers1)


df = pd.read_csv('SPECT.train', names=headers1)
print(df.shape)
print(list(df))

names = ['positive', 'negative']
values = df['overall_diagnosis'].value_counts(normalize=True)
test_df = pd.read_csv('SPECT.test', names=headers1)
test_values = test_df['overall_diagnosis'].value_counts(normalize=True)

plt.figure(5, figsize=(10,10), dpi= 20, facecolor='w', edgecolor='k')
plt.subplot(111)
plt.bar(names, values)
plt.suptitle('Figure 1 : Model')
plt.savefig('model_poisson.png', dpi=128)
plt.close()

plt.figure(5, figsize=(10,10), dpi=20, facecolor='w', edgecolor='g')
plt.subplot(111)
plt.bar(names, test_values)
plt.suptitle('Figure 2 : Evaluation')
plt.savefig('evaluation_poisson.png', dpi=128)
plt.close()

chi2_stat, p_val, dof, ex = stat.chi2_contingency(test_values)

print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
print(ex)
