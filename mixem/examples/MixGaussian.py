#!/usr/bin/env python
import numpy as np
import mixem
from mixem.distribution import MultivariateNormalDistribution
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA

def mixGaussian(n,mean,cov,weights):
	assert sum(weights) == 1
	category = np.flatnonzero(np.random.multinomial(1,weights))
	cmean = mean[category,:][0]
	return category, np.random.multivariate_normal(cmean,cov)

def recover(data):
    
    mu = [np.mean(data[0,:]),np.mean(data[1,:])]
    sigma = [np.var(data[0,:]),np.var(data[1,:])]
    #print mu,sigma

    init_params = [
        (np.array((mu[0] - 1, mu[0] + 1)), np.identity(2)),
        (np.array((mu[1] - 1, mu[1] + 1)), np.identity(2)),
    ]

    start = time.time()
    
    weight, distributions, ll, iteration = mixem.em(data, [MultivariateNormalDistribution(mu, sigma) for mu, sigma in init_params])

    #print(weight, distributions, ll)
    #print 'iterate time: ' + str(time.time() - start) + ' seconds'

    return weight,distributions,iteration,(time.time() - start)

if __name__ == '__main__':
    n = 2
    sample_number = 300
    mean = np.array([[1,1],[-1,-1]])
    cov = np.array([[2,0.5],[0.5,1]])
#********** Judge Covariance spherical or ellipsoidal*
    w,v = LA.eig(cov)

    weights = np.array([0.3,0.7])
    data = np.zeros((sample_number,2))
      
    count = np.zeros((sample_number,2))
    test_count = 100
    iteration = np.zeros((test_count,1))
    time_cost = np.zeros((test_count,1))
    quality = np.zeros((test_count,1))
    success = 0

    for j in range(test_count):
	original_first = np.zeros((sample_number,2))
	original_second = np.zeros((sample_number,2))
	first_index = 0
	second_index = 0
	for i in range(sample_number):
	    count[i],data[i] = mixGaussian(n,mean,cov,weights)
	    #print data[i]
	    if np.count_nonzero(count[i]) == 2:
		original_first[first_index] = data[i]
		first_index += 1
	    else:
		original_second[second_index] = data[i]
		second_index += 1

#***** Compute the original data's mean and Cov ***	
	original_weights = [np.count_nonzero(count)/(sample_number*2.0) , 1.0 - np.count_nonzero(count)/(sample_number*2.0)]
	original_first.resize((first_index,2))
	original_second.resize((second_index,2))

	first_mu = [np.mean(original_first[0,:]),np.mean(original_first[1,:])]
	second_mu = [np.mean(original_second[0,:]),np.mean(original_second[1,:])]
	
#************Now Using the EM ***********
	weight,distribution,iteration[j],time_cost[j] = recover(data)
	weight = np.array(weight)
	first_mu = np.array(first_mu)
	second_mu = np.array(second_mu)
	original_weights = np.array(original_weights)
	if (weight[0] < 0.4 and weight[0] > 0.2) and distribution[0].mu[0] > 0:
	    quality[j] += LA.norm([original_weights[0]-weight[1],original_weights[1]-weight[0]]) + LA.norm(distribution[0].mu - second_mu) + LA.norm(distribution[1].mu - first_mu) + LA.norm(distribution[0].sigma - np.cov(original_second.transpose())) + LA.norm(distribution[1].sigma - np.cov(original_first.transpose()))

	    success += 1
	elif (weight[0] < 0.8 and weight[0] > 0.6) and distribution[0].mu[0] < 0:
	    success += 1
	    quality[j] += LA.norm(original_weights-weight) + LA.norm(distribution[0].mu - first_mu) + LA.norm(distribution[1].mu - second_mu) + LA.norm(distribution[0].sigma - np.cov(original_first.transpose())) + LA.norm(distribution[1].sigma - np.cov(original_second.transpose()))
	else:
	    quality[j] = 0
	    iteration[j] = 0
	    time_cost[j] = 0

    print "Iteration counts: " , np.sum(iteration)/np.count_nonzero(iteration)
    print "Time consuming: " , np.sum(time_cost)/np.count_nonzero(time_cost)
    print "Quality (The larger the worse): ", np.sum(quality)/np.count_nonzero(quality)
    print ("Success time {0} in {1}".format(success,test_count))
    print "The eigenvalues: " ,w



