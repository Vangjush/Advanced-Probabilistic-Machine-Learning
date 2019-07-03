# This code is mainly the same as http://edwardlib.org/tutorials/supervised-regression
# However, comments have been added and the code includes some small modifications.
# Please send any comments to: pekka.marttinen@aalto.fi
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

plt.style.use('ggplot')


# CREATE DATA

np.random.seed(0)

def build_toy_dataset(N, w):
    D = len(w)
    x = np.random.normal(0.0, 2.0, size=(N,D))
    y = np.dot(x, w) + np.random.normal(0.0, 0.01, size=N)
    return x, y

N = 200
D = 10

w_true = np.random.randn(D) * 0.5 # so, the true STD of weights is 0.5
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)



# DEFINE MODEL

X = tf.placeholder(tf.float32, [N, D])
# Model is conditional on RV X, therefore this is not a distribution (unlike for y). This is not a tf.Variable either, because we are not learning a value for X, but rather we assign it a value below. We could have assigned it here already, but have decided to do it 'on the fly', which is possible when it is defined as a 'placeholder'.


# The prior STD of the weights is defined using a tf.constant.
w_prior_std = tf.constant(1.0)


# Note: w and b are here random variables, and they are assigned prior distributions. Contrast this with parameters (defined using tf.Variables) and input data X (defined as a placeholder). The posterior disribution of w and b will be estimated below.
w = Normal(loc=tf.zeros(D), scale=w_prior_std * tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))
# y is also a random variable with a distribution, unlike X, which is a placeholder. This is because the supervised regression model is conditional on X, i.e., X is not modeled per se, but y is modeled conditional on X.



# DEFINE VARIATIONAL DISTRIBUTION

qw = Normal(loc=tf.Variable(tf.random_normal([D])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
# qw is a vector of D random variables with independent Normal distributions for its elements. These distributions are the 'factors' which are optimized in the VB algorithm, and they are parameterized using 'variational parameters' (the tf.Variables). The variational parameters themselves don't have a distribution, but a single fixed value is learned for them, and that's why they must be represented as tf.Variables (similarly to the 'w_prior_std' above).

# One more variational factor for b:
qb = Normal(loc=tf.Variable(tf.random_normal([1])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))



# INFERENCE

inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
# This approximates the posterior of w and b using factors qw and qb.
# Here a value 'X_train' is fed to the placeholder X. y is also given a value y_train. Note that y is actually the 'data' that we are modeling here (i.e. it has a distribution), whereas X is not modeled per se (but the distribution of y is conditional on X).
# NOTE: inference defined in this way will automatically optimize the ELBO with respect to all tf.Variables. In this case this means that fixed values are learned not only for all the variational parameters but also for the 'w_prior_std'. The learned values can be recovered by running w_prior_std.eval() in the end.


inference.run(n_samples=5, n_iter=250)
# n_samples: Number of samples from variational model for calculating stochastic gradients
# n_iter: Number of iterations when calling run(), default=1000
# NB: these values seem sometimes not sufficient for convergence. More accurate inference may be obtained, at least with some data sets, with default values using just: inference.run() without any arguments.



# MODEL CRITICISM

y_post = ed.copy(y, {w: qw, b: qb})
# Same as:
# y_post = Normal(loc=ed.dot(X, qw) + qb, scale=noise_std * tf.ones(N))
# In other words, this is the posterior predictive distribution for y.

print("Mean squared error:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))
# evaluate can take an additional argument 'n_samples', which specifies the number of samples from the posterior when making predictions, using the posterior predictive distribution

print('Mean absolute error:')
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))



# VISUALIZATION
def visualise(X_data, y_data, w, b, ax, n_samples=10):
    w_samples = w.sample(n_samples)[:, 0].eval()
    b_samples = b.sample(n_samples).eval()
    ax.scatter(X_data[:, 0], y_data) # Note, only the 1st input dimension is plotted.
    inputs = np.linspace(-8, 8, num=400)
    for ns in range(n_samples):
        output = inputs * w_samples[ns] + b_samples[ns]
        ax.plot(inputs, output)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

visualise(X_train, y_train, w, b, ax1) # Models sampled from the prior
visualise(X_train, y_train, qw, qb, ax2) # Models sampled from the posterior
plt.show()



# EXPLORE THE LEARNED MODEL

print('Point estimate for STD of weights:', w_prior_std.eval())

# Retrieve the means and STDs of the estimated regression coefficients
w_est_mean = qw.mean().eval()
w_est_std = qw.stddev().eval()
print('Correlation between estimated and learned weights: ', np.corrcoef(w_est_mean, w_true)[0,1])
