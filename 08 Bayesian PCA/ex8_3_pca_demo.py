
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

np.random.seed(123)
plt.close('all')
plt.style.use('ggplot')

def build_toy_dataset(N, D, K, sigma=1.3):
    x_train = np.zeros((D, N))
    w = np.random.normal(0.0, 2.0, size=(D, K))
    z = np.random.normal(0.0, 1.0, size=(K, N))
    mean = np.dot(w, z)
    for d in range(D):
        for n in range(N):
            x_train[d, n] = np.random.normal(mean[d, n], sigma)

    print('True principal axes')
    print(w)
    return x_train

N = 5000
D = 2 # Data dimension
K = 1 # Latent dimension

x_train = build_toy_dataset(N, D, K)


# MODEL
noise_std = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
w = Normal(loc=tf.zeros([D, K]), scale=2.0 * tf.ones([D, K])) # prior on w
z = Normal(loc=tf.zeros([N, K]), scale=tf.ones([N, K])) # prior on z 
x = Normal(loc=tf.matmul(w, z, transpose_b=True), scale=tf.ones([D, N]) * noise_std) # likelihood
# transpose_b=True transposes the second argument


# INFERENCE
qw = Normal(loc=tf.Variable(tf.random_normal([D, K])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qz = Normal(loc=tf.Variable(tf.random_normal([N, K])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([N, K]))))

inference = ed.KLqp({w: qw, z: qz}, data={x: x_train}) # Note: noise std is not updated
inference.run(n_iter=500, n_print=100, n_samples=10)


# CRITICISM
print('Inferred principal axes:')
print(qw.mean().eval())

x_post = ed.copy(x, {w: qw, z: qz}) # Simulate x_post similarly to x, but use learned z and w
x_gen = x_post.sample().eval()

print('Inferred noise_std')
print(noise_std.eval())

# VISUALIZATION
def visualise(x_data, y_data, ax, color, title):
    ax.scatter(x_data, y_data, color=color, alpha=0.1)
    ax.axis([-10, 10, -10, 10])
    ax.set_title(title)
    
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

visualise(x_train[0, :], x_train[1, :], ax1, 'blue', 'Observed')
visualise(x_gen[0, :], x_gen[1, :], ax2, 'red', 'Posterior')

plt.show()

#plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1)
#plt.axis([-10, 10, -10, 10])
#plt.title('Simulated data set')
#plt.show()


    
