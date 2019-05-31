# libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
sample_mean = []

for iteration in range(100):
	df_subsample = df.sample(1000)
	mean = df_subsample['Price'].mean()
	sample_mean.append(mean)


plt.hist(sample_mean, bins=10, density=True, histtype='step')
population_mean = df['Price'].mean()
population_variance = df['Price'].std()
print('Population mean:', population_mean)
print('Population variance:', population_variance)

mean_sample_mean = np.mean(sample_mean)
mean_sample_variance = np.std(sample_mean)
print(mean_sample_variance)

difference_mean = df['Price'].mean() - np.mean(sample_mean)
print(difference_mean)

plt.title('Histogram of Sample Mean Price')
plt.savefig('out.png', dpi=128)
plt.close()





































