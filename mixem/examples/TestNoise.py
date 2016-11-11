#!/usr/bin/env python
import matplotlib.pyplot as plt
import os
from numpy import linalg as LA
import numpy as np
import pandas as pd
import math
import mixem
from mixem.distribution import MultivariateNormalDistribution

def main():

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "faithful.csv"))

    data = np.array(data)

    init_params = [
        (np.array((2, 50)), np.identity(2)),
        (np.array((4, 80)), np.identity(2)),
    ]
# ************ Noise - EM , The first boolean value means noise or not ************
#**************The second value means the fraction ***********


    
    test_time = 50
    time_consuming_n = np.zeros((test_time,1))
    iterations = np.zeros((test_time,1))
    quality = np.zeros((test_time,1))

    gather_time_consuming = np.zeros((21,1))
    gather_iterations = np.zeros((21,1))
    gather_quality = np.zeros((21,1))
    index = 0
    fraction = 0
    while fraction < 1.05:
	#noise = True
	if fraction  == 0:
	    noise = False
	else:
	    noise = True
	for k in range(test_time):
	    data1 = np.copy(data)
	    weight, distributions, ll, time_consuming_n[k],iterations[k] = mixem.Nem(noise,fraction,data1, [MultivariateNormalDistribution(mu, sigma) for mu, sigma in init_params], initial_weights=[0.3, 0.7])

	    quality[k] += LA.norm(weight[0] * distributions[0].mu + weight[1] * distributions[1].mu-np.array(np.mean(data[:,0]),np.mean(data[:,1]))) + LA.norm(weight[0] * distributions[0].sigma + weight[1] * distributions[1].sigma - np.cov(data.transpose()))

        if noise == True:
	    print "With {0} noise, time_consuming: {1}".format(fraction,np.mean(time_consuming_n))
    	    print "With {0} noise, quality: {1}".format(fraction,np.mean(quality))
    	    print "With {0} noise, iteration count : {1}".format(fraction,np.mean(iterations))
        else:
	    print "Without noise, time_consuming: ",np.mean(time_consuming_n)
	    print "Without noise, quality: ",np.mean(quality)
	    print "Without noise, iteration: ",np.mean(iterations)
	gather_time_consuming[index] = np.mean(time_consuming_n)
	gather_quality[index] = np.mean(quality)
	gather_iterations[index] = np.mean(iterations)
	index += 1
        fraction += 0.05
    x_axis = np.arange(0.,1.05,0.05)
    plt.figure(1)
    plt.subplot(311)
    plt.plot(x_axis,gather_time_consuming,'ro')
    plt.title("Red:Average Time Convergence\n Blue: Quality   Green: Iteration counts")
    #plt.xlabel("Fraction of Standard Deviation")
    plt.ylabel("Times (seconds)")

    plt.subplot(312)
    plt.plot(x_axis,gather_quality,'b+')
    #plt.title("Quality (Using Norm) of Each Noise")
    #plt.xlabel("Fraction of Standard Deviation")
    plt.ylabel("Qualitys")

    plt.subplot(313)
    plt.plot(x_axis,gather_iterations,'g*')
    #plt.title("Iteration Counts of Each Noise")
    plt.xlabel("Fraction of Standard Deviation")
    plt.ylabel("Iteration Counts")
    plt.show()
if __name__ == '__main__':
    main()
