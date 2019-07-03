import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, psi

np.random.seed(123123120)


def compute_stochastic_elbo_gradient(alpha_tau, beta_tau, r1, r2, m2, beta2, alpha0, beta0, x, n_simulations):
	n_data_items = len(r1)
	
	# Derivatives w.r.t. alpha_tau, beta_tau, m2, beta2, and all r1 terms.
	elbo_grad_array = np.zeros((n_simulations, 4 + n_data_items))
	
	
	for simu_index in range(n_simulations):
		# Estimate the gradient by sampling (i.e. simulating) from the current
		# approximation n_simulations many times, and average in the end.
		
		
		# SAMPLE unobservables from the current approximation
		tau_s, theta_s, z_s = sample_from_q(alpha_tau, beta_tau, r1, r2, m2, beta2)
		
		# COMPUTE MODEL LOG JOINT:
		# log[p(tau)] + log[p(theta)] + log[p(z|tau)] + log[p(x|z,theta)]
		
		# log[Beta(tau|alpha0,alpha0)]
		log_p_tau = gammaln(2 * alpha0) - 2 * gammaln(alpha0) + (alpha0 - 1) * np.log(tau_s) + (alpha0 - 1) * np.log(1-tau_s)
		
		# log[N(theta|0,beta0^(-1))
		log_p_theta = 0.5 * np.log(beta0) - 0.5 * np.log(2*np.pi) - 0.5 * beta0 * theta_s ** 2
		
		# log[p(z|tau)]
		N1 = np.sum(z_s[:,0])
		N2 = np.sum(z_s[:,1])
		log_p_z_cond_tau = N1 * np.log(1-tau_s) + N2 * np.log(tau_s)
		
		# log[p(x|z,theta)]
		N = N1 + N2
		log_p_x_cond_z_theta = -0.5 * N * np.log(2*np.pi) - 0.5 * np.sum(z_s[:,0] * x ** 2) - 0.5 * np.sum(z_s[:,1] * (x - theta_s) ** 2)
		
		log_joint_p = log_p_tau + log_p_theta + log_p_z_cond_tau + log_p_x_cond_z_theta
		
		
		# COMPUTE LOG JOINT OF APPROXIMATION Q:
		# log[q(tau)] + log[q(theta)] + log[q(z)]
		
		log_q_tau = ?     # EXERCISE, note: the gammaln function has been imported from scipy.special
		
		log_q_theta = ?   # EXERCISE
		
		log_q_z = ?       # EXERCISE
		
		log_joint_q = log_q_tau + log_q_theta + log_q_z
		
		
		# COMPUTE GRADIENT of loq[q(tau,theta,z)] w.r.t. variational parameters
		
		d_alpha_tau = ?   # EXERCISE, note: the psi function has been imported from scipy.special
		
		d_beta_tau = ?    # EXERCISE
		
		d_m2 = ?          # EXERCISE
		
		d_beta2 = ?       # EXERCISE
		
		d_rn1 = ?         # EXERCISE
		
		# COMBINE EVERYTHING to form the gradient of the ELBO:
		elbo_grad_array[simu_index, :] = np.concatenate([[d_alpha_tau, d_beta_tau, d_m2, d_beta2], d_rn1]) * (log_joint_p - log_joint_q)
	
	
	# AVERAGE over the samples:
	elbo_grad = np.mean(elbo_grad_array, axis=0)
	
	return elbo_grad


def sample_from_q(alpha_tau, beta_tau, r1, r2, m2, beta2):
	tau_s = ?    # EXERCISE
	theta_s = ?  # EXERCISE
	
	z1 = ?  # EXERCISE
	z2 = ?  # EXERCISE
	
	# z_s contains z1 and z2 as its columns
	z_s = np.array([z1, z2]).T
	
	return tau_s, theta_s, z_s


# Compute ELBO for the model described in simple_elbo.pdf
def compute_elbo(alpha_tau, beta_tau, r1, r2, m2, beta2, alpha0, beta0, x):
    
    # E[log p(tau)]
    term1 = (alpha0 - 1) * (psi(alpha_tau) + psi(beta_tau) - 2 * psi(alpha_tau + beta_tau))

    # E[log p(theta)]
    term2 = -0.5 * beta0 * (beta2**(-1) + m2**2)

    # E[log p(z|tau)]
    N2 = np.sum(r2); N1 = np.sum(r1); N = N1 + N2
    term3 = N2 * psi(alpha_tau) + N1 * psi(beta_tau) - N * psi(alpha_tau + beta_tau)

    # E[log p(x|z,theta)]
    term4 = -0.5 * np.sum(r1 * x**2) - 0.5 * np.sum(r2 * ((x-m2)**2 + beta2**(-1)))

    # Negative entropy of q(z)
    term5 = np.sum(r1 * np.log(r1)) + np.sum(r2 * np.log(r2))

    # Negative entropy of q(tau)
    term6 = (gammaln(alpha_tau + beta_tau) - gammaln(alpha_tau) - gammaln(beta_tau)
        + (alpha_tau - 1) * psi(alpha_tau) + (beta_tau - 1) * psi(beta_tau)
        - (alpha_tau + beta_tau - 2) * psi(alpha_tau + beta_tau))

    # Negative entropy of q(theta)
    term7 = 0.5 * np.log(beta2)

    elbo = term1 + term2 + term3 + term4 - term5 - term6 - term7
    
    return elbo




# Simulate data
theta_true = 4
tau_true = 0.3
n_samples = 50
z = (np.random.rand(n_samples) < tau_true)  # True with probability tau_true
x = np.random.randn(n_samples) + z * theta_true


# Parameters of the prior distributions.
alpha0 = 1.5
beta0 = 1

n_iter = 600
# To keep track of the estimates of tau and theta in different iterations:
tau_est = np.zeros(n_iter)
th_est = np.zeros(n_iter)
elbo_array = np.zeros(n_iter)   # To track the elbo


# Some initial values for the variational parameters
alpha_tau = 1
beta_tau = 1
beta_2 = 1
m2 = 1
r1 = np.random.rand(n_samples)   # Responsibilities of the first cluster.
r2 = 1 - r1

for it in range(n_iter):
	step_size = 0.02 / (10+it)**0.5
	
	# Compute the gradient of the ELBO
	elbo_grad = compute_stochastic_elbo_gradient(alpha_tau, beta_tau, r1, r2, m2, beta_2, alpha0, beta0, x, 200)
	
	# Update factor q(tau) using stochastic gradient
	alpha_tau = np.max([alpha_tau + step_size * elbo_grad[0], 0.1])
	beta_tau  = np.max([beta_tau  + step_size * elbo_grad[1], 0.1])
	
	# Update factor q(theta) using stochastic gradient
	m2 = m2 + step_size * elbo_grad[2]
	beta_2 = beta_2 + step_size * elbo_grad[3]
	
	# Update responsibilites, factor q(z), using closed-form updates
	E_log_tau = psi(alpha_tau) - psi(alpha_tau + beta_tau)
	E_log_tau_c = psi(beta_tau) - psi(alpha_tau + beta_tau)
	E_log_var = (x-m2)**2 + 1/beta_2
	
	log_rho1 = E_log_tau_c - 0.5 * np.log(2*np.pi) - 0.5 * (x**2)
	log_rho2 = E_log_tau - 0.5 * np.log(2*np.pi) - 0.5 * E_log_var
	max_log_rho = np.maximum(log_rho1, log_rho2)   # Normalize to avoid numerical problems when exponentiating.
	rho1 = np.exp(log_rho1 - max_log_rho)
	rho2 = np.exp(log_rho2 - max_log_rho)
	r2 = rho2 / (rho1 + rho2)
	r1 = 1 - r2
	
	# Keep track of the current estimates
	tau_est[it] = (alpha_tau) / (alpha_tau + beta_tau)
	th_est[it] = m2
	
	# Compute the ELBO
	elbo_array[it] = compute_elbo(alpha_tau, beta_tau, r1, r2, m2, beta_2, alpha0, beta0, x)
	
	print("Iteration %i: theta=%f, tau=%f" % (it, m2, tau_est[it]))
	# With large enough n_samples, this should eventually converge 
	# to (theta_true, tau_true), at least approximately.


plt.plot(elbo_array)
plt.title('ELBO')
plt.show()
