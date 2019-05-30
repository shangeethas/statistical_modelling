# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import factorial

df = pd.read_csv('Absenteeism_at_work.csv', delimiter=';')

plt.plot(df['Age'], df['Absenteeism time in hours'], 'bo')
plt.xlabel('Age')
plt.ylabel('Absenteeism time in hours')
plt.title('Absenteeism at Work')
plt.savefig('age.png', dpi=128)
plt.close()

plt.plot(df['Distance from Residence to Work'], df['Absenteeism time in hours'], 'r+')
plt.xlabel('Distance from Residence to Work')
plt.ylabel('Absenteeism time in hours')
plt.title('Absenteeism at Work')
plt.savefig('distance.png', dpi=128)
plt.close()

plt.hist(df['Age'], bins=20, density=True)
plt.xlabel('Age of Employees')
plt.ylabel('Probability wrt Total Employees')
plt.title('Histogram of Age Factor of Employees')
plt.savefig('histogram_age.png', dpi=128)
plt.close()

plt.hist(df['Distance from Residence to Work'], bins=20, density=True)
plt.xlabel('Distance from Residence to Work')
plt.ylabel('Probability wrt Total Employees')
plt.title('Histogram of Distance Factor of Employees')
plt.savefig('histogram_distance.png', dpi=128)
plt.close()

plt.hist(df['Service time'], bins=5, density=True)
plt.xlabel('Service Time')
plt.ylabel('Probability wrt Total Employees')
plt.title('Histogram of Service Time Factor of Employees')
plt.savefig('histogram_service_time.png', dpi=128)
plt.close()

plt.hist(df['Body mass index'], bins=5, density=True)
plt.xlabel('Body mass index')
plt.ylabel('Probability wrt Total Employees')
plt.title('Histogram of BMI Factor of Employees')
plt.savefig('histogram_bmi.png', dpi=128)
plt.close()


def likelihood(theta, n, x):
    return (factorial(n) / (factorial(x) * factorial(n - x))) * (theta ** x) * ((1 - theta) ** (n - x))


n = 10.
x = 7.
prior = x / n

possible_theta_values = list(map(lambda x: x / 100, range(100)))

likelihoods = list(map(lambda theta: likelihood(theta, n, x), possible_theta_values))

mle = possible_theta_values[np.argmax(likelihoods)]

f, ax = plt.subplots(1)
ax.plot(df['Age'], df['Absenteeism time in hours'])
ax.axvline(mle, linestyle="--")
ax.set_xlabel("Theta")
ax.set_ylabel("Likelihood")
ax.grid()
ax.set_title('Likelihood of Age')
plt.close()
