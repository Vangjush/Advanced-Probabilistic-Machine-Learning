import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm

# initialize the random number generator
np.random.seed(24)



# SIMULATE THE TRUE DATA SET
num_samples = 5   # Try e.g. values between 5 and 5000
mu_true = 2       # mean of the true distribution
lambda_true = 4   # precision of the true distribution
sigma_true = 1 / np.sqrt(lambda_true)    # standard deviation
data_set = np.random.normal(mu_true, sigma_true, num_samples)

# plot the data set as a histogram
plt.xlim([0, 5])
plt.hist(data_set, np.arange(0.125, 4, 0.25), rwidth=0.9)
plt.title("Histogram of $%i$ samples from $N(%s, %s)$" % (num_samples, mu_true, sigma_true))
plt.show()



# SPECIFY PRIORS

# lambda is the precision parameter of the unknown Gaussian
# and it is given a prior distribution Gamma(a0,b0),
# (a0 is the 'shape' and b0 the 'rate')
a0 = 0.01
b0 = 0.01   # These correspond to a noninformative prior

# mu is the mean parameter of the unknown Gaussian
# and it is given a prior distribution that depends on
# lambda: N(mu0, (beta0*lambda)^-1)
mu0 = 0
beta0 = 0.001   # Low precision corresponds to high variance

# (This is the so-called Normal-Gamma(mu0, beta0, a0, b0) prior distribution for mu and lambda)



# LEARN THE POSTERIOR DISTRIBUTION

# Due to conjugacy, the posterior distribution is also Normal-Gamma(mu_n, beta_n, a_n, b_n)

sample_mean = np.mean(data_set)
sample_var = np.var(data_set)

mu_n = (mu0 * beta0 + num_samples * sample_mean) / (beta0 + num_samples)

beta_n = beta0 + num_samples

a_n = a0 + num_samples / 2

b_n = b0 + (num_samples * sample_var + (beta0 * num_samples * (sample_mean - mu0) ** 2)
           / (beta0 + num_samples)) / 2



# PLOT THE PRIOR AND THE POSTERIOR DISTRIBUTIONS

# Plot distribution of lambda, the precision
lambda_range = np.arange(0, 10, 0.01)
prior_lambda_pdf = gamma.pdf(lambda_range, a0, scale=1/b0)
posterior_lambda_pdf = gamma.pdf(lambda_range, a_n, scale=1/b_n)

plt.plot(lambda_range, prior_lambda_pdf, label="prior")
plt.plot(lambda_range, posterior_lambda_pdf, label="posterior")
plt.plot([lambda_true,lambda_true], [0,1], "k-", label="true value")
plt.title('lambda')
plt.legend()
plt.show()

# Plot distribution of mu, the mean
mu_range = np.arange(1, 3, 0.01)

# Because mu depends on lambda, we need to integrate over lambda.
# We do this by Monte Carlo integration (i.e. average over multiple simulated lambdas)
def mu_pdf_monte_carlo(a, b, mu, beta):
	gamma_samples = np.random.gamma(a, 1/b, 100)
	sum_mu_pdf = np.zeros(len(mu_range))
	for gamma_sample in gamma_samples:
		mu_pdf = norm.pdf(mu_range, mu, 1 / np.sqrt((beta * gamma_sample)))
		sum_mu_pdf += mu_pdf
	mu_pdf = sum_mu_pdf / len(gamma_samples)
	return mu_pdf

prior_mu_pdf     = mu_pdf_monte_carlo(a0,  b0,  mu0,  beta0)
posterior_mu_pdf = mu_pdf_monte_carlo(a_n, b_n, mu_n, beta_n)

plt.plot(mu_range, prior_mu_pdf, label="prior")
plt.plot(mu_range, posterior_mu_pdf, label="posterior")
plt.plot([mu_true,mu_true],[0,2.5], "k-", label="true value")
plt.title('mu')
plt.legend()
plt.show()


# PLOT THE TRUE AND ESTIMATED DISTRIBUTIONS OF THE SAMPLES

# We estimate the parameters with the mean of the posterior distribution
mu_hat = np.sum(posterior_mu_pdf * mu_range) / np.sum(posterior_mu_pdf)
lambda_hat = np.sum(posterior_lambda_pdf * lambda_range) / np.sum(posterior_lambda_pdf)

full_dist_range = np.arange(-2, 6, 0.1)
true_pdf = norm.pdf(full_dist_range, mu_true, sigma_true)
estimated_pdf = norm.pdf(full_dist_range, mu_hat, 1 / np.sqrt(lambda_hat))

plt.plot(full_dist_range, true_pdf, label="true")
plt.plot(full_dist_range, estimated_pdf, label="estimated")
plt.title('Distribution of the samples')
plt.legend()
plt.show()


# COMPUTE K-L DIVERGENCE BETWEEN TRUE AND ESTIMATED SAMPLE DISTRIBUTIONS

# Hints for the exercise:
# For computing the KL-divergence, use numerical integration over a grid of
# values. "full_dist_range" specifies a suitable grid along the x-axis.
# Values of the true PDF estimated at the grid points are given in 
# "true_pdf" and values of the estimated PDF at the grid points are given
# in "estimated_pdf". For computing the integral, you can use any numerical
# integration available in Numpy, e.g., the "trapz" function.

KL = true_pdf 
