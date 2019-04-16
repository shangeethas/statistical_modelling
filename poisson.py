#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat

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

plt.hist(df['overall_diagnosis'], bins=50, density=True)
plt.xlabel('Probability')
plt.ylabel('No of bins')
plt.title('Histogram of Training Data Set')
#plt.axvline(x=df['overall_diagnosis'].mean(), color='red')
print(df['overall_diagnosis'].mean())
plt.savefig('actual_out_poisson.png', dpi=128)
plt.close()

test_df	= pd.read_csv('SPECT.test', names=headers1)
print(test_df.shape)
plt.hist(test_df['overall_diagnosis'], bins=50, density=True)
plt.xlabel('Probability')
plt.ylabel('No of bins')
plt.title('Histogram of Test Data Set')
#plt.axvline(x=df['overall_diagnosis'].mean(), color='red')
print(test_df['overall_diagnosis'].mean())
print(stat.chisquare(test_df))
plt.savefig('expected_out_poisson.png', dpi=128)
plt.close()

#chi_square=(expected-actual)2/expected
