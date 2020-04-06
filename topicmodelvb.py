"""
Fit truncated-HDP and NNFA-HDP topic models using SVI. 
"""

import sys, re, time, string
import numpy as n
from scipy.special import logsumexp

import matplotlib.pyplot as plt

from utils import *
from corpus import *

n.random.seed(100000001)
meanchangethresh = 0.001

class _TopicModel:
    """
    Parent class for SVI training of topic models (NNFA-HDP or T-HDP).
    """

    def __init__(self, vocab, K, T, topicpath, D, omega, alpha, eta, tau0, kappa):
        """
        Arguments:
            K: Number of topics i.e. corpus-level truncation
            T: Max number of topics manifested by a document i.e. document-level truncation
            vocab: A set of words to recognize. When analyzing documents, any word
               not in this set will be ignored.
            D: Total number of documents in the population. 
            topicpath: Path to some pre-trained topics' i.e. variational Dirichlet parameters. 
            omega: corpus-level stick-breaking Beta(1,omega)
            alpha: per-document stick-breaking Beta(1,alpha)
            eta: Hyperparameter for prior on topics beta
            tau0: A (positive) learning parameter that downweights early iterations
            kappa: Learning rate: exponential decay rate---should be between
                 (0.5, 1.0] to guarantee asymptotic convergence.
        Remarks:
            The inheritted classes T_HDP and NNFA_HDP each have additional instance
            variables for the global variational parameters.
            User should make sure the topics from topicpath are compatible with 
            K and vocab.
        """
        self._vocab, self._idxtoword = make_vocab(vocab)

        self._K = K
        self._T = T
        self._W = len(self._vocab)
        self._D = D
        self._omega = omega
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0
        
        # for updating per-document stick-breaking
        mask = n.zeros((self._T, self._T))
        for i in range(self._T):
            for j in range(self._T):
                mask[i,j] = int(j > i)
        self._Tmask = mask # size (self._T, self._T)
        
        # for updating topic-to-topic
        mask = n.zeros((self._K, self._K))
        for i in range(self._K):
            for j in range(self._K):
                mask[i,j] = int(j > i)
        self._Kmask = mask # size (self._K, self._K)
        
        return

    def log_likelihood_one(self, wordobs_ids, wordobs_cts, wordho_ids, \
                      wordho_cts):
        """
        Inputs:
            wordobs_ids: list, index in vocab of unique observed words
            wordobs_cts: list, number of occurences of each unique observed word
            wordho_ids: list, index in vocab of held-out words
            wordho_cts: list, number of occurences of each unique held-out word
        Outputs:
            average log-likelihood of held-out words for the given document
        """
        # theta_means should be 1 x self._K
        theta_means = self.theta_means(wordobs_ids, wordobs_cts)
        # lambda_sums should be self._K x 1
        lambda_sums = n.sum(self._lambda, axis=1) 
        # lambda_means should be self._K x self._W, rows suming to 1
        lambda_means = self._lambda/lambda_sums[:, n.newaxis] 
        Mho = list(range(0,len(wordho_ids)))
        proba = [wordho_cts[i]*n.log(n.dot(theta_means,lambda_means[:,wordho_ids[i]])) \
                for i in Mho]
        # average across all held-out words
        tot = sum(wordho_cts)
        return sum(proba)/tot

    def log_likelihood_docs(self, wordids, wordcts):
        """
        Inputs:
            wordids: list of lists (unique ids in the documents)
            wordcts: list of lists (count of words in the documents)
        Outputs:
        """ 
        t0 = time.time()
        M = len(wordids)
        log_likelihoods = []
        for i in range(M):
            docids = wordids[i] # list 
            doccts = wordcts[i] # list
            # only evaluate log-likelihood if non-trivial document
            if len(docids) > 1:
                wordobs_ids, wordobs_cts, wordho_ids, wordho_cts = \
                    split_document(docids, doccts)
                doc_likelihood = \
                    self.log_likelihood_one(wordobs_ids, wordobs_cts, wordho_ids, wordho_cts)
                log_likelihoods.append(doc_likelihood)
        t1 = time.time()
        # print("Time taken to evaluate log-likelihood %.2f" %(t1-t0))
        return n.mean(log_likelihoods)
    
class T_HDP(_TopicModel):
    """
    Inherit _TopicModel to train truncated HDP. 
    """

    def __init__(self, vocab, K, T, topicpath, D, omega, alpha, eta, tau0, kappa):
        """
        Arguments:
            K: Number of topics i.e. corpus-level truncation
            T: Max number of topics manifested by a document i.e. document-level truncation
            vocab: A set of words to recognize. When analyzing documents, any word
               not in this set will be ignored.
            D: Total number of documents in the population. 
            topicpath: Path to some pre-trained topics' i.e. variational Dirichlet parameters. 
            omega: corpus-level stick-breaking Beta(1,omega)
            alpha: per-document stick-breaking Beta(1,alpha)
            eta: Hyperparameter for prior on topics beta
            tau0: A (positive) learning parameter that downweights early iterations
            kappa: Learning rate: exponential decay rate---should be between
                 (0.5, 1.0] to guarantee asymptotic convergence.
        Remarks:
            The inheritted classes T_HDP and NNFA_HDP might use more 
            instance variables. 
            User should make sure the topics from topicpath are compatible with 
            K and vocab.
        """
        t0 = time.time()
        
        # Common instance variables
        _TopicModel.__init__(self, vocab, K, T, topicpath, D, omega, alpha, eta, tau0, kappa)
        
        # Truncation representation of the corpus-level DP
        ## Initialize q(beta|lambda) if topicpath is not None
        if (topicpath is None):
            self._lambda = np.random.gamma(1.0, 1.0, (self._K, self._W)) * self._D*100/(self._K*self._W)
        else:
            self._lambda = n.loadtxt(topicpath)
            assert self._lambda.shape == (self._K,self._W), "Wrong shape of topics"
            print("Successfully loaded topics from %s" %topicpath)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        ## Initialize a and b using the prior
        self._a = n.ones(self._K)
        self._b = self._omega*n.ones(self._K)
        self._b[-1] = 0
        
        t1 = time.time()
        print("Time to initialize topic model using %d topics is %.2f" %(self._K, t1-t0))
        return
    
    def do_m_step(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        topics lambda and the corpus-level stick-breaking a and b.

        Arguments:
            wordids: list of lists
            wordcts: list of lists
    
        Returns:
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update zeta, phi, gamma | lambda, a, b for this
        # mini-batch. This also returns the information about phi, zeta that
        # we need to update lambda, a, b
        varparams, sstats = self.do_e_step(wordids, wordcts)
        # Update global parameters
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats["lambda"] / len(wordids))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._a = self._a * (1-rhot) + \
            rhot * (1 + self._D * sstats["a"] / len(wordids))
        self._a[-1] = 1
        self._b = self._b * (1-rhot) + \
            rhot * (self._omega + self._D * sstats["b"] / len(wordids))
        self._b[-1] = 0 
        self._updatect += 1

        return varparams
    
    def init_doc(self, ids, cts):
        """
        Inputs:
            ids: unique word indices
            cts: number of occurences of words in ids
        Outputs:
            initialize topic-to-topic and word-assignment using data but 
            stick-breaking using the prior
        """
        Elogbetad = self._Elogbeta[:, ids] # (self._K, len(ids)) 
        ## zeta. Calculate in log-domain since summation across words lead 
        ## to underflow
        tempzetad = n.dot(Elogbetad, cts) 
        logzetadnorm = logsumexp(tempzetad)
        initzetad = n.exp(tempzetad-logzetadnorm) # shape (self._K,)
        # zeta's shape: (self._T, self._K)
        zetad = n.multiply(n.ones((self._T, self._K)), initzetad[n.newaxis, :]) 
        
        ## phi
        # phi's shape: (self._T, len(ids))
        tempphid = n.dot(zetad, Elogbetad)
        logphidnorm = logsumexp(tempphid, axis=0) # shape (len(ids),)
        logphid = tempphid - logphidnorm[n.newaxis,:] # (self._T, len(ids))
        # normalize across rows 
        phid = n.exp(logphid)
        
        ## gamma
        gamma1d = n.ones(self._T)
        gamma2d = self._alpha*n.ones(self._T)
        gamma2d[-1] = 0
        
        return (zetad, phid, gamma1d, gamma2d)
    
    def do_e_step(self, wordids, wordcts):
        """
        Inputs:
            wordids: list of ids
            wordcts: list of cts
        Outputs:
            sstats
        """
        batchD = len(wordids)
    
        # topic-to-topic and stick-breaking are sufficient to evaluate 
        # held-out log-likelihood
        zeta = n.zeros((batchD, self._T, self._K))
        gamma1 = n.zeros((batchD, self._T))
        gamma2 = n.zeros((batchD, self._T))
        
        sstats = {}
        sstats["lambda"] = n.zeros(self._lambda.shape)
        sstats["a"] = n.zeros(self._a.shape)
        sstats["b"] = n.zeros(self._b.shape)
        
        it = 0
        meanchange = 0
        converged = False
        # Now, for each document d, update that document's topic-to-topic, 
        # stick-breaking and word assignents
        for d in range(0, batchD):
            ids = wordids[d]
            cts = wordcts[d]
            Elogbetad = self._Elogbeta[:, ids]
            
            # initialize local variational parameters
            zetad, phid, gamma1d, gamma2d = self.init_doc(ids, cts)
            
            for it in range(0, 200):
                lastzetad = zetad 
                lastphid = phid
                lastgamma1d = gamma1d
                lastgamma2d = gamma2d
                
                # stick-breaking update
                gamma1d = 1 + n.dot(phid, cts)
                gamma1d[-1] = 1
                gamma2d = self._alpha + n.dot(n.dot(self._Tmask, phid), cts)
                gamma2d[-1] = 0                
                
                # topic-to-topic update
                """
                print(self._a)
                print(self._b)
                """
                Elogtheta = expect_log_sticks(self._a, self._b, self._Kmask) # shape (self._K,)
                # By default, n.multiply(phid, cts) multiplies every column of phid 
                # by the same number in cts 
                tempzetad = n.dot(n.multiply(phid, cts), Elogbetad.transpose()) # shape (self._T, self._K)
                tempzetad = tempzetad + Elogtheta[n.newaxis,:]
                logzetadnorm = logsumexp(tempzetad, axis=1)
                logzetad = tempzetad - logzetadnorm[:, n.newaxis]
                zetad = n.exp(logzetad)
                
                # word assignment update
                Elogpid = expect_log_sticks(gamma1d, gamma2d, self._Tmask) # shape (self._T,)
                tempphid = n.dot(zetad, Elogbetad) # shape (self._T, len(ids))
                tempphid = tempphid + Elogpid[:, n.newaxis]
                logphidnorm = logsumexp(tempphid, axis=0)
                logphid = tempphid - logphidnorm[n.newaxis, :]
                phid = n.exp(logphid)
                
                # If parameters didn't change much, we're done.
                meanchange = 0.25*(n.mean(abs(gamma1d - lastgamma1d)) + n.mean(abs(gamma2d - lastgamma2d)) + \
                        n.mean(abs(zetad - lastzetad)) + n.mean(abs(phid - lastphid)))
                
                if (meanchange < meanchangethresh):
                    converged = True
                    break
            
            # Store topic-to-topic and stick-breaking
            zeta[d,:,:] = zetad
            gamma1[d,:] = gamma1d
            gamma2[d,:] = gamma2d
            
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats["lambda"][:, ids] += n.multiply(n.dot(zetad.transpose(), phid), cts)
            sstats["a"] += (n.sum(zetad,axis=0)).flatten()
            sstats["b"] += n.sum(n.dot(self._Kmask, zetad.transpose()), axis=1)

        return (zeta, gamma1, gamma2), sstats
    
    def checkpoint_e_step(self, wordids, wordcts):
        """
        Inputs:
            wordids: list of ids
            wordcts: list of cts
        Outputs:
            sstats
        """
        batchD = len(wordids)
    
        # topic-to-topic and stick-breaking are sufficient to evaluate 
        # held-out log-likelihood
        zeta = n.zeros((batchD, self._T, self._K))
        gamma1 = n.zeros((batchD, self._T))
        gamma2 = n.zeros((batchD, self._T))
        
        sstats = {}
        sstats["lambda"] = n.zeros(self._lambda.shape)
        sstats["a"] = n.zeros(self._a.shape)
        sstats["b"] = n.zeros(self._b.shape)
        
        it = 0
        meanchange = 0
        converged = False
        # Now, for each document d, update that document's topic-to-topic, 
        # stick-breaking and word assignents
        for d in range(0, batchD):
            ids = wordids[d]
            cts = wordcts[d]
            Elogbetad = self._Elogbeta[:, ids]
            
            # initialize local variational parameters
            zetad, phid, gamma1d, gamma2d = self.init_doc(ids, cts)
            
            for it in range(0, 200):
                lastzetad = zetad 
                lastphid = phid
                lastgamma1d = gamma1d
                lastgamma2d = gamma2d
                
                # stick-breaking update
                gamma1d = 1 + n.dot(phid, cts)
                gamma1d[-1] = 1
                gamma2d = self._alpha + n.dot(n.dot(self._Tmask, phid), cts)
                gamma2d[-1] = 0                
                
                # topic-to-topic update
                """
                print(self._a)
                print(self._b)
                """
                Elogtheta = expect_log_sticks(self._a, self._b, self._Kmask) # shape (self._K,)
                # By default, n.multiply(phid, cts) multiplies every column of phid 
                # by the same number in cts 
                tempzetad = n.dot(n.multiply(phid, cts), Elogbetad.transpose()) # shape (self._T, self._K)
                tempzetad = tempzetad + Elogtheta[n.newaxis,:]
                logzetadnorm = logsumexp(tempzetad, axis=1)
                logzetad = tempzetad - logzetadnorm[:, n.newaxis]
                zetad = n.exp(logzetad)
                
                # word assignment update
                Elogpid = expect_log_sticks(gamma1d, gamma2d, self._Tmask) # shape (self._T,)
                tempphid = n.dot(zetad, Elogbetad) # shape (self._T, len(ids))
                tempphid = tempphid + Elogpid[:, n.newaxis]
                logphidnorm = logsumexp(tempphid, axis=0)
                logphid = tempphid - logphidnorm[n.newaxis, :]
                phid = n.exp(logphid)
                
                # If parameters didn't change much, we're done.
                meanchange = 0.25*(n.mean(abs(gamma1d - lastgamma1d)) + n.mean(abs(gamma2d - lastgamma2d)) + \
                        n.mean(abs(zetad - lastzetad)) + n.mean(abs(phid - lastphid)))
                
                if (meanchange < meanchangethresh):
                    converged = True
                    break
            
            # Store topic-to-topic and stick-breaking
            zeta[d,:,:] = zetad
            gamma1[d,:] = gamma1d
            gamma2[d,:] = gamma2d
            
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats["lambda"][:, ids] += n.multiply(n.dot(zetad.transpose(), phid), cts)
            sstats["a"] += (n.sum(zetad,axis=0)).flatten()
            sstats["b"] += n.sum(n.dot(self._Kmask, zetad.transpose()), axis=1)

        return (zeta, gamma1, gamma2), sstats

    def theta_means(self, wordobs_ids, wordobs_cts):
        """
        Inputs:
            wordobs_ids = list
            wordobs_cts = list
        Outputs:
            Report E(q(theta(k)), where q(theta) is variational 
            approximation of the document's topic proportions w.r.t the corpus-level
            topics, computing using both topic-to-topic and stick-breaking. 
        """
        # do E-step for the wordobs_ids portion of the document
        varparams, _ = self.do_e_step([wordobs_ids],[wordobs_cts]) 
        zeta = varparams[0][0] # shape (self._T, self._K)
        gamma1 = varparams[1][0] # shape (self._T,)
        gamma2 = varparams[2][0] # shape (self._T,)
        
        sndlevel = GEM_expectation(gamma1[n.newaxis,:], gamma2[n.newaxis,:], self._T) # shape (1, self._T)
        thetad = n.dot(sndlevel, zeta) # shape (1, self._K)
        
        return thetad

def sanity_E_step(seed, K, topicpath):
    """
    Examine effect of SB-LDA E-step's on a document
    Inputs:
        seed: seed for replicability
        K: number of topics
        topicpath: file path of pre-trained topics for warm-start training
    Outputs:
    """
    ## load topics
    inroot = "wiki10k"
    infile = inroot + "_wordids.csv"
    with open(infile) as f:
        D = sum(1 for line in f)
    vocab = open('./dictnostops.txt').readlines()
    tm = SB_LDA(vocab, K, topicpath, D, 1, 0.01, 1024., 0.7)
    
    ## E-step on a document, plotting initial guess of topic proportions 
    ## as well as their convergence, but make no updates to underlying topics 
    n.random.seed(seed)
    (wordids, wordcts) = \
            get_batch_from_disk(inroot, D, 6)
    fig1, axes1 = plt.subplots(2,3,figsize=(12,8))
    fig2, axes2 = plt.subplots(2,3,figsize=(12,8))
    for i in range(6):
        cidx = i % 3
        ridx = i // 3
        _, elbo1, oldbound1, changes1, sstats1 = tm.debug_e_step(wordids[i], wordcts[i], "tau")
        _, elbo2, oldbound2, changes2, sstats2 = tm.debug_e_step(wordids[i], wordcts[i], "phi")
        # ELBO
        axes1[ridx, cidx].plot(elbo1, marker='o', markersize=2, label="tau")
        axes1[ridx, cidx].plot(elbo2, marker='o', markersize=2, label="phi")
        axes1[ridx, cidx].set_xlabel("Iteration")
        axes1[ridx, cidx].set_ylabel("ELBO")
        axes1[ridx, cidx].set_title("Document %i contains %d words" %(i, sum(wordcts[i])))
        axes1[ridx, cidx].legend()
        # change in tau
        axes2[ridx, cidx].plot(changes1, marker='o', markersize=2, label="tau")
        axes2[ridx, cidx].plot(changes2, marker='o', markersize=2, label="phi")
        axes2[ridx, cidx].set_xlabel("Iteration")
        axes2[ridx, cidx].set_ylabel("Mean change in tau")
        axes2[ridx, cidx].set_title("Document %i contains %d words" %(i, sum(wordcts[i])))
        axes2[ridx, cidx].legend()
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
    return

def sanity_M_step(seed, K, topicpath):
    ## load topics
    inroot = "wiki10k"
    infile = inroot + "_wordids.csv"
    with open(infile) as f:
        D = sum(1 for line in f)
    vocab = open('./dictnostops.txt').readlines()
    tm = SB_LDA(vocab, K, topicpath, D, 1, 0.01, 1024., 0.7)
    
    ## Do 10 M-steps based on 10 sampled documents. All steps 
    ## print out the change in ELBO as function of topics after 
    ## the update: should always be positive!
    n.random.seed(seed)
    (wordids, wordcts) = \
            get_batch_from_disk(inroot, D, 10)
    for i in range(10):
        _ = tm.debug_update_lambda(wordids[i], wordcts[i])
    return

def main():
    sanity_E_step(12, 100, "results/sbldaK100_D16_wiki10k_wiki1k/lambda-100.dat")
    sanity_M_step(1, 100, None)
    return

if __name__ == '__main__':
    main()