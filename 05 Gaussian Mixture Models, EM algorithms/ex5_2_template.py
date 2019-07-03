import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


### Simulate data:

np.random.seed(0)

theta_true = 3
tau_true = 0.5
n_samples = 100

x = np.zeros(n_samples)
for i in range(n_samples):
	# Sample from N(0,1) or N(theta_true,1)
	if np.random.rand() < 1 - tau_true:
		x[i] = np.random.normal(0, 1)
	else:
		x[i] = np.random.normal(theta_true, 1)


### The EM algorithm:

n_iter = 20
theta = np.zeros(n_iter)
tau = np.zeros(n_iter)

# Initial guesses for theta and tau
theta[0] = 1
tau[0] = 0.1

for it in range(1, n_iter):
	# The current estimates for theta and tau,
	# computed in the previous iteration
	theta_0 = theta[it-1]
	tau_0 = tau[it-1]
	
	# E-step: compute the responsibilities r1 and r2
	r1 = # EXERCISE
	r2 = # EXERCISE
	
	# M-step: compute the parameter values that maximize
	# the expectation of the complete-data log-likelihood.
	theta[it] = # EXERCISE
	tau[it] = # EXERCISE


# Print and plot the values of theta and tau in each iteration
print("theta       tau")
for theta_i, tau_i in zip(theta, tau):
	print("{0:.7f}  {1:.7f}".format(theta_i, tau_i))

plt.plot(range(n_iter), theta, label = 'theta')
plt.plot(range(n_iter), tau, label = 'tau')
plt.xlabel('Iteration')
plt.legend()
plt.show()
