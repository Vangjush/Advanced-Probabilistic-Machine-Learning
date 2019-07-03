import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import GMMem
import pickle

np.random.seed(0)

# Fill in the missing parts marked with ?


# ***** DATA

totalComponents = 10  # max number of mixture components

# load data
with open("data.pickle", "rb") as f:
	X, labels = pickle.load(f)

D, N = X.shape   # dimension and number of data points

ratio = 0.75
train_ind = np.random.choice(N, int(ratio * N), replace=False)   # training data index
test_ind = np.setdiff1d(np.arange(N), train_ind)                 # test data index

Xtrain = X[:,train_ind]            # training data
Xtrain_labels = labels[train_ind]  # training data labels

Xtest = X[:,test_ind]            # test data
Xtest_labels = labels[test_ind]  # test data labels

# plot training and test data
def plot_data():
	for i in sorted(set(Xtrain_labels)):
		X_comp = Xtrain[:, Xtrain_labels == i]
		plt.plot(X_comp[0], X_comp[1], '.' + 'brgmcyk'[i-1], markersize=6)
	
	plt.plot(Xtest[0], Xtest[1], 'kd', markersize=4, markeredgewidth=0.5, markerfacecolor="None")

plot_data()
plt.title('training data, test data (in black)')
plt.show()



# ***** Use BIC to select the number of components
# (Only this part differs from the second template, where cross validation is used instead)

BICs = []

for H in range(1, totalComponents+1):    # number of mixture components
	print("H: {}".format(H))
	
	P, m, S, loglik, phgn = GMMem.GMMem(Xtrain, H, 100)  # fit to data
	
	numParams = ?     # number of parameters in the model
	BIC = ?           # BIC for the model
	BICs.append(BIC)


# plot the BIC curve
plt.bar(range(1, totalComponents+1), BICs)
plt.yscale("log", nonposy="clip")
plt.xlabel('Number of Mixture Components')
plt.ylabel('BIC')
plt.title('Model Selection (BIC)')
plt.show()

# select the number of mixture components which minimizes the BIC
h = np.argmin(BICs) + 1



# ***** TRAIN

# Now train full model with selected number of mixture components
P, m, S, loglik, phgn = GMMem.GMMem(Xtrain, h, 100)  # fit to data

# Predict using the full trained model (Use GMMem.GMMloglik)
logl = ?

print('Test Data Likelihood = {0:f}'.format(?))


# Plot the best GMM model
plot_data()

for i in range(h):
	dV, E = LA.eig(S[i,:,:])
	
	theta = np.arange(0, 2*np.pi, 0.1)
	p = np.sqrt(dV.reshape(D,1)) * [np.cos(theta), np.sin(theta)]
	x = (E @ p) + np.tile(m[:,i:i+1], (1, len(theta)))
	
	plt.plot(x[0], x[1], 'r-', linewidth=2)

plt.title('training data, test data (in black)')
plt.show()
