# libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from scipy.stats import norm

df = pd.read_csv('Absenteeism_at_work.csv', delimiter=';')
print(df.shape)
n = df.shape[0]
mu = df['Service time'].mean()
sigma = df['Service time'].std()

print('Likelihood mu:', mu)
print('Likelihood sigma:', sigma)

# Likelihood
fig, ax = plt.subplots(1, 1)
x = np.linspace(0, 30, 200)
likelihood = norm.pdf(x,mu,sigma)
ax.plot(x, st.norm.pdf(x, mu, sigma),'r-', lw=2, alpha=0.6, label='Likelihood')


r = df['Service time']
ax.hist(r, normed=True, bins=5, histtype='stepfilled')
ax.set_xlabel('Service Time in Years')
ax.set_ylabel('Probability density')
ax.set_title('Bayesian Inference for Service Time')

# personal prior
prior_mu = 10
prior_sigma = 2
prior = norm.pdf(x, prior_mu, prior_sigma)
ax.plot(x, prior, 'b-', lw=2, label='Prior')

# posterior
posterior = prior * likelihood
ax.plot(x, posterior, 'k-', lw=2, label='Posterior')

plt.legend()
plt.savefig('likelihood_service_time.png', dpi=128)
plt.close()
















