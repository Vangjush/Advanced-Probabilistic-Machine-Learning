import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


### Simulate data:

np.random.seed(0)

theta_true = 3
n_samples = 100

x = np.zeros(n_samples)
for i in range(n_samples):
	# Sample from N(0,1) or N(theta_true,1) with equal probability
	if np.random.rand() < 0.5:
		x[i] = np.random.normal(0, 1)
	else:
		x[i] = np.random.normal(theta_true, 1)

# Plot the data
plt.hist(x, bins=25)
plt.show()


### The EM algorithm:

n_iter = 20
theta = np.zeros(n_iter)

# Initial guess for theta
theta[0] = 0

for it in range(1, n_iter):
	# The current estimate for theta,
	# computed in the previous iteration
	theta_0 = theta[it-1]
	
	# E-step: compute the responsibilities r2 for component 2
	r1_unnorm = scipy.stats.norm.pdf(x, 0, 1)
	r2_unnorm = scipy.stats.norm.pdf(x, theta_0, 1)
	r2 = r2_unnorm / (r1_unnorm + r2_unnorm)
	
	# M-step: compute the parameter value that maximizes
	# the expectation of the complete-data log-likelihood.
	theta[it] = sum(r2 * x) / sum(r2)


# Print and plot the values of theta in each iteration
print("theta")
for theta_i in theta:
	print("%.7f" % theta_i)

plt.plot(range(n_iter), theta)
plt.show()
