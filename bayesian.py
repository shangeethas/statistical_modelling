#libraries
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

def likelihood(theta, n, x):
    return (factorial(n) / (factorial(x) * factorial(n - x))) \
            * (theta ** x) * ((1 - theta) ** (n - x))

n = 10.
x = 7.
prior = x / n

possible_theta_values = list(map(lambda x: x/100, range(100)))


likelihoods = list(map(lambda theta: likelihood(theta, n, x)\
                                , possible_theta_values))

mle = possible_theta_values[np.argmax(likelihoods)]

f, ax = plt.subplots(1)
ax.plot(possible_theta_values, likelihoods)
ax.axvline(mle, linestyle = "--")
ax.set_xlabel("Theta")
ax.set_ylabel("Likelihood")
ax.grid()
ax.set_title("Likelihood of Theta")
plt.show()
