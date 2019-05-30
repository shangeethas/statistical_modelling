#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat

df = pd.read_csv('aflstats/stats.csv')
print(df.shape)

is_adelaide = df['Team']=='Adelaide'
print(is_adelaide.head())
df_adelaide = df[is_adelaide]
print(df_adelaide.shape)

is_atkins_rory = df_adelaide['Player']=='Atkins, Rory'
print(is_atkins_rory.head())
df_atkins_rory = df_adelaide[is_atkins_rory]
print(df_atkins_rory.shape)


plt.hist(df_atkins_rory['Goals'], bins=100, density=True)
plt.title('Histogram of Goals scored by Atkins Rory')
plt.xlabel('No of Goals')
plt.ylabel('No of Matches')
plt.savefig('out_goals.png', dpi=128)
plt.close()

plt.hist(df_atkins_rory['PercentPlayed'], bins=20, density=True)
plt.title('Histogram of Percentage Played by Atkins Rory')
plt.xlabel('Percentage Played')
plt.savefig('out_percentage_played.png', dpi=128)
plt.close()



