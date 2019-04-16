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

plt.hist(df['Price'], bins=200, density=True, histtype='step')
plt.hist(sample_mean, bins=10, density=True, histtype='step')
plt.axvline(x=np.mean(sample_mean))
plt.axvline(x=df['Price'].mean(), color="red")
plt.ylabel('Probability')
plt.savefig('out.png', dpi=128)
plt.close()
















   






































