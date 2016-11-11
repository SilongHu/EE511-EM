import numpy as np
import time
import mixem
import math

def Nem(Noise,fraction,data, distributions, initial_weights=None, max_iterations=1000, tol=1e-15, tol_iters=1, progress_callback=mixem.simple_progress):
    """Fit a mixture of probability distributions using the Expectation-Maximization (EM) algorithm.

    :param data: The data to fit the distributions for. Can be an array-like or a :class:`numpy.ndarray`
    :type data: numpy.ndarray

    :param distributions: The list of distributions to fit to the data.
    :type distributions: list of :class:`mixem.distribution.Distribution`

    :param initial_weights:  Inital weights for the distributions. Must be the same size as distributions. If None, will use uniform initial weights for all distributions.
    :type initial_weights: numpy.ndarray

    :param max_iterations:  The maximum number of iterations to compute for.
    :type max_iterations: int

    :param tol: The minimum relative increase in log-likelihood after tol_iters iterations
    :type tol: float

    :param tol_iters: The number of iterations to go back in comparing log-likelihood change
    :type tol_iters: int

    :param progress_callback: A function to call to report progress after every iteration.
    :type progress_callback: function or None

    :rtype: tuple (weights, distributitons, log_likelihood)
    """

    np.seterr(divide = 'ignore', invalid = 'ignore')
    n_distr = len(distributions)
    n_data = data.shape[0]

    if initial_weights is not None:
        weight = np.array(initial_weights)
    else:
        weight = np.ones((n_distr,))

    last_ll = np.zeros((tol_iters, ))
    resp = np.empty((n_data, n_distr))
    log_density = np.empty((n_data, n_distr))
    iteration = 0
    start = time.time()
    more = 0
    #if fraction != 0:
	#noise_X = np.random.normal(0,fraction * np.std(data[:,0]),data.shape[0])
	#noise_Y = np.random.normal(0,fraction * np.std(data[:,1]),data.shape[0])
    #else:
	#noise_X = 0
	#noise_Y = 0
    #noise_x = np.copy(noise_X)
    #noise_y = np.copy(noise_Y)
    #print np.mean(noise_X),np.mean(data[:,0])
    #print np.mean(noise_Y),np.mean(data[:,1])
    #data1 = np.zeros((data.shape[0],2))
    data1 = np.copy(data)
    #data2 = np.copy(original_data)
    #noise_X = data1[:,0] - data2[:,0] 
    #noise_Y = data1[:,1] - data2[:,1] 
    


    #data[:,0] =  data[:,0] + noise_X
    #data[:,1] =  data[:,1] + noise_Y
    if (Noise == True):
	#noise_x = np.copy(noise_X)
	#noise_y = np.copy(noise_Y)
	while True:
        # E-step #######
	    addtime = time.time()
	    noise_x = np.random.normal(0,fraction * np.std(data[:,0])/(iteration + 1)**2,data.shape[0]) 
	    noise_y = np.random.normal(0,fraction * np.std(data[:,1])/(iteration + 1)**2,data.shape[0])
	    #print "In function: ",noise_x
	    data1[:,0] += noise_x
	    data1[:,1] += noise_y
	    more += (time.time() - addtime)
        # compute responsibilities
	    #print n_distr
            for d in range(n_distr):
		#log_density(data)
		#print len(data2.shape)
            	log_density[:, d] = distributions[d].log_density(data1)
	    #print log_density
	    for k in range(np.shape(log_density)[0]):
		for p in range(np.shape(log_density)[1]):
		    if log_density[k][p] < -500:
			log_density[k][p] = -500
	    #print np.shape(log_density)[0]
	
	    #print "Log: ",log_density
	    #print np.exp(log_density)
	    #print weight[np.newaxis,:]
        # normalize responsibilities of distributions so they sum up to one for example
            resp = weight[np.newaxis, :] * np.exp(log_density)
	    #resp = weight * np.exp(log_density)
	    #print "resp" , resp
            resp /= np.sum(resp, axis=1)[:, np.newaxis]
	    #resp /= 1
	    #print resp
	    #for i in range(np.shape(resp)[0]):
		#if math.isnan(resp[i][0]):
		    #resp[i][0] = 0
		    #resp[i][1] = 0
            log_likelihood = np.sum(resp * log_density)
	    #print log_likelihood
        # M-step #######
            for d in range(n_distr):
            	distributions[d].estimate_parameters(data1, resp[:, d])

            weight = np.mean(resp, axis=0)

            #if progress_callback:
            	#progress_callback(iteration, weight, distributions, log_likelihood)

        # Convergence check #######
            if np.isnan(log_likelihood):
            	last_ll[0] = log_likelihood
                break

            if iteration >= tol_iters and (last_ll[-1] - log_likelihood) / last_ll[-1] <= tol:
            	last_ll[0] = log_likelihood
            	break

            if iteration >= max_iterations:
            	last_ll[0] = log_likelihood
            	break

        # store value of current iteration in last_ll[0]
        # and shift older values to the right
            last_ll[1:] = last_ll[:-1]
            last_ll[0] = log_likelihood

            iteration += 1
        #print "Iteration time of EM with Noise: " + str(time.time() - start - more) + " seconds"
	#print iteration
        return weight, distributions, last_ll[0], time.time() - start - more,iteration

    elif (Noise == False):
	while True:
        # E-step #######
        # compute responsibilities
            for d in range(n_distr):
            	log_density[:, d] = distributions[d].log_density(data1)
        # normalize responsibilities of distributions so they sum up to one for example
            resp = weight[np.newaxis, :] * np.exp(log_density)
	    #print resp
            resp /= np.sum(resp, axis=1)[:, np.newaxis]
	    #print resp
	    #for i in range(np.shape(resp)[0]):
		#if math.isnan(resp[i][0]):
		    #resp[i][0] = 0
		    #resp[i][1] = 0
	    
            log_likelihood = np.sum(resp * log_density)
	    #print log_likelihood
        # M-step #######
            for d in range(n_distr):
            	distributions[d].estimate_parameters(data1, resp[:, d])

            weight = np.mean(resp, axis=0)

            #if progress_callback:
            	#progress_callback(iteration, weight, distributions, log_likelihood)

        # Convergence check #######
            if np.isnan(log_likelihood):
            	last_ll[0] = log_likelihood
                break

            if iteration >= tol_iters and (last_ll[-1] - log_likelihood) / last_ll[-1] <= tol:
            	last_ll[0] = log_likelihood
            	break

            if iteration >= max_iterations:
            	last_ll[0] = log_likelihood
            	break

        # store value of current iteration in last_ll[0]
        # and shift older values to the right
            last_ll[1:] = last_ll[:-1]
            last_ll[0] = log_likelihood

            iteration += 1
        return weight, distributions, last_ll[0], time.time() - start,iteration
    else:
	print "Noise Error!"
	return 0,0,0
