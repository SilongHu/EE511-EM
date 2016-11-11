#!/usr/bin/env python

import numpy as np
import mixem
from mixem.distribution import MultivariateNormalDistribution
import time

def generate_data():
    dist_params = [
        (np.array([4]), np.diag([1])),
        (np.array([1]), np.diag([0.5]))
    ]
#    print np.cov(dist_params)
    weights = [0.3, 0.7]

    n_data = 1000
    data = np.zeros((n_data, 1))
    first = []
    second = []
    for i in range(n_data):
        dpi = np.random.choice(range(len(dist_params)), p=weights)
        dp = dist_params[dpi]
        data[i] = np.random.multivariate_normal(dp[0], dp[1])
	if dpi == 0:
	    first.append(data[i])
	else:
	    second.append(data[i])
    print len(first),len(second)
    return data

def recover(data):
    
    mu = np.mean(data)
    sigma = np.var(data)
    init_params = [
        (np.array([mu + 0.1]), np.diag([sigma])),
        (np.array([mu - 0.1]), np.diag([sigma]))
    ]
 #   start = time.time()
    weight, distributions, ll = mixem.em(data, [MultivariateNormalDistribution(mu, sigma) for mu, sigma in init_params])

    print(weight, distributions, ll)
#    print 'iterate time: ' + str(time.time() - start) + ' seconds'


if __name__ == '__main__':
    data = generate_data()
    recover(data)
