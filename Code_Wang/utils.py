#! /usr/bin/python

''' several useful functions '''
import numpy as np

def log_normalize(v):
    ''' return log(sum(exp(v)))'''

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:,np.newaxis]

    return (v, log_norm)

def log_sum(log_a, log_b):
	''' we know log(a) and log(b), compute log(a+b) '''
	v = 0.0;
	if (log_a < log_b):
		v = log_b+np.log(1 + np.exp(log_a-log_b))
	else:
		v = log_a+np.log(1 + np.exp(log_b-log_a))
	return v


def argmax(x):
	''' find the index of maximum value '''
	n = len(x)
	val_max = x[0]
	idx_max = 0

	for i in range(1, n):
		if x[i]>val_max:
			val_max = x[i]
			idx_max = i		

	return idx_max			


def GEM_expectation(tau1, tau2):
    """
    Inputs:
        tau1: 1 x K, positive numbers, last number is 1 
        tau2: 1 x K, non-negative numbers, last number is 0
    Outputs:
        E(theta(k)) where theta(k) = Beta(tau1(k), tau2(k)) prod{i=1}{k-1} (1-Beta(tau1(k), tau2(k)))
    """
    # theta(k) = p(k) x prod_{i=1}^{k-1} (1-p(i)), each p(i) Beta(tau1(i), tau2(i))
    # and they are independent because of mean-field.
    Ep = tau1/(tau1+tau2)
    Em1p = 1-Ep # last value is 0 since theta(K) is Beta(1,0)
    Em1p[0,-1] = 1 # hack
    cumu = np.cumprod(Em1p, axis=1) # shape (K,)
    ratiop = Ep/Em1p # shape (1, K)
    theta = np.multiply(ratiop, cumu) # shape (1, K)
    return theta