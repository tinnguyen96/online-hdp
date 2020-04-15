"""
Fit LDA or T-dSB-DP or N-dSB-DP using SVI. 
"""

import sys, re, time, string
import numpy as n
from scipy.special import logsumexp
from scipy.stats import entropy

import matplotlib.pyplot as plt

from utils import *
from corpus import *

n.random.seed(100000001)
meanchangethresh = 0.001

class LDA:
    """
    SVI training of topic models LDA.
    """

    def __init__(self, vocab, K, topicfile, D, alpha0, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. 
        topicfile: Dictionary containing path to some other topic model's output topics and
                the name of the topic model.
        alpha0: for LDA 1/K, alpha = alpha0/K is the hyperparameter for prior on topic 
            proportions theta_d. For SB-LDA, alpha0 is governs the stick-breaking
            Beta(1,alpha0). 
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Remarks:
            User should make sure the topics from topicpath are compatible with 
            K and vocab.
        """
        
        t0 = time.time()
        self._alpha = alpha0/K
        self._vocab, self._idxtoword = make_vocab(vocab)

        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda) if topicpath is not None
        if (topicfile is None):
            self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        else:
            self._lambda = n.loadtxt(topicfile["lambda"])
            assert self._lambda.shape==(self._K,self._W), "Wrong shape of topics"
            print("Successfully loaded topics from %s" %topicfile["lambda"])
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        t1 = time.time()
        print("Time to initialize topic model using %d topics is %.2f" %(self._K, t1-t0))
        
        return
    
    def save_topics(self, savedir, iteration):
        """Save topics"""
        lambdaname = (savedir + "/lambda-%d.dat") % iteration
        n.savetxt(lambdaname, self._lambda)
        return 
    
    def do_e_step(self, wordids, wordcts):
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        # each gamma[:,:] has mean 1 and variance 0.01
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        converged = False
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 200):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    converged = True
                    break
            # might have exited coordinate ascent without convergence
            """
            if (not converged):
                print("Coordinate ascent in E-step didn't converge")
                print("Last change in gammad %f" %meanchange)
            """
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return (gamma,sstats) 

    def theta_means(self, wordobs_ids, wordobs_cts):
        """
        Inputs:
            wordobs_ids = list
            wordobs_cts = list
        Outputs:
            Report E(q(theta(k)) across topics, where q(theta) is variational 
            approximation of the new document's topic proportions.
        """
        # do E-step for the document represented by the observed words
        # gamma should be 1 x self._K
        gamma, _ = self.do_e_step([wordobs_ids],[wordobs_cts]) 
        # q(theta|gamma) is Dirichlet, so marginal means are average of Dirichlet parameters
        theta = gamma/n.sum(gamma) 
        theta = theta.flatten(order='C') 
        return theta

    def update_lambda(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.
    
        Returns variational parameters for per-document topic proportions.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        varparams, sstats = self.do_e_step(wordids, wordcts)
        # Estimate held-out likelihood for current values of lambda.
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / len(wordids))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return varparams

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
            wordids: list of lists
            wordcts: list of lists
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
    
class _TopicModel:
    """
    Parent class for SVI training of topic models (N-dSB-DP or T-dSB-DP).
    """

    def __init__(self, vocab, K, T, topicfile, D, omega, alpha, eta, tau0, kappa):
        """
        Arguments:
            K: Number of topics i.e. corpus-level truncation
            T: Max number of topics manifested by a document i.e. document-level truncation
            vocab: A set of words to recognize. When analyzing documents, any word
               not in this set will be ignored.
            D: Total number of documents in the population. 
            topicfile: Dictionary containing path to some other topic model's output topics and
                the name of the topic model. 
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
    
    def check_invariant(self, zetad, phid, tol=1e-10):
        """
        Inputs:
            zetad: (self._T, self._K) topic-to-topic
            phid: (self._T, len(ids)) topic assignments
            tol: scalar, tolerance level
        Outputs:
            rowsums of zetad = 1
            colsums of phid = 1
        """
        zinv = n.abs(n.sum(zetad, axis=1)-1) < tol
        pinv = n.abs(n.sum(phid, axis=0)-1) < tol
        return (n.all(zinv) and n.all(pinv))
    
    def init_doc(self, ids, cts, debug=False):
        """
        Inputs:
            ids: unique word indices
            cts: number of occurences of words in ids
            debug: whether to check for invariants
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
        # zeta's shape: (self._T, self._K). The rows are identical.
        zetad = n.multiply(n.ones((self._T, self._K)), initzetad[n.newaxis, :]) 

        ## phi (implicitly defined)
        # phi's shape: (self._T, len(ids))
        tempphid = n.dot(zetad, Elogbetad)
        logphidnorm = logsumexp(tempphid, axis=0) # shape (len(ids),)
        logphid = tempphid - logphidnorm[n.newaxis,:] # (self._T, len(ids))
        # normalize across rows. The columns are identical.
        phid = n.exp(logphid)

        ## gamma
        gamma1d = n.ones(self._T)
        gamma2d = self._alpha*n.ones(self._T)
        gamma2d[-1] = 0
        
         # check invariants
        if (debug):
            invariant = self.check_invariant(zetad, phid)
            if (not invariant):
                print("Something wrong with initialization")

        return (zetad, phid, gamma1d, gamma2d)
    
    def L1(self, zetad, phid, gamma1d, gamma2d, ids, cts):
        """
        Compute ELBO as function of document's variational parameters.
        Inputs:
            zetad, phid, gamma1d, gamma2d: current values of local variational
                params. 
                zetad: (self._T, self._K)
                phid: (self._T, len(ids))
                gamma1d: (self._T,)
                gamma2d: (self._T,)
            ids, cts: unique word indices and their number of occurences in 
                the doc
        Outputs: ELBO
        """
        # word-dependent terms
        ## Elogbeta_{c_{d,z_{dn}}(w_{dn})
        Elogbetad = self._Elogbeta[:,ids] # (self._K, len(ids))
        Elogbetacw = n.sum(n.multiply(phid, n.dot(zetad, Elogbetad)),axis=0) # (len(ids),)
        ## Elogp(z_{dn}|pi_d)
        Elogpid = expect_log_sticks(gamma1d, gamma2d) # (self._T,)
        phiElogpid = n.dot(phid.transpose(), Elogpid) # (len(ids),)
        ## entropy
        ent = entropy(phid) # (len(ids),)
        perword = Elogbetacw + phiElogpid + ent
        term1 = n.sum(n.multiply(perword, cts))
        
        # stick-breaking terms 
        kls = beta_KL(gamma1d[:(self._T-1)], gamma2d[:(self._T-1)], 1, self._alpha)
        term2 = -n.sum(kls)
        
        # topic-to-topic terms
        Elogtheta = self._Elogtheta # (self._K,)
        ent = n.sum(entropy(zetad.transpose()))
        term3 = n.dot(Elogtheta, n.sum(zetad, axis=0)) + ent
        
        return term1 + term2 + term3
    
    def do_e_step(self, wordids, wordcts, getelbo=False, debug=False):
        """
        Inputs:
            wordids: list of ids
            wordcts: list of cts
            getelbo: whether to report ELBO
            debug: whether to check invariants of local params
        Outputs:
            varparams: optimal local variational parameters
            sstats: sufficient statistics for global parameters update
            ELBO: dictionary, keys being documents, values being ELBO versus 
                coordinate ascent iteration for that doc
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
        
        if (getelbo):
            ELBO = {}
        else:
            ELBO = None

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
            zetad, phid, gamma1d, gamma2d = self.init_doc(ids, cts, debug)       
        
            if (getelbo):
                ELBO[d] = []
                initELBO = self.L1(zetad, phid, gamma1d, gamma2d, ids, cts)
                ELBO[d].append(initELBO)
            
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
                # By default, n.multiply(phid, cts) multiplies every column of phid 
                # by the same number in cts 
                tempzetad = n.dot(n.multiply(phid, cts), Elogbetad.transpose()) # shape (self._T, self._K)
                tempzetad = tempzetad + self._Elogtheta[n.newaxis,:]
                logzetadnorm = logsumexp(tempzetad, axis=1)
                logzetad = tempzetad - logzetadnorm[:, n.newaxis]
                zetad = n.exp(logzetad)
                
                # word assignment update
                Elogpid = expect_log_sticks(gamma1d, gamma2d) # shape (self._T,)
                tempphid = n.dot(zetad, Elogbetad) # shape (self._T, len(ids)). Potentially the most time-consuming part of E-step
                tempphid = tempphid + Elogpid[:, n.newaxis]
                logphidnorm = logsumexp(tempphid, axis=0)
                logphid = tempphid - logphidnorm[n.newaxis, :]
                phid = n.exp(logphid)
                
                # evaluate ELBO in getelbo mode
                if (getelbo):
                    ELBO[d].append(self.L1(zetad, phid, gamma1d, gamma2d, ids, cts))
                
                # check if local params' invariants hold in debug mode
                if (debug):
                    invariant = self.check_invariant(zetad, phid)
                    if (not invariant):
                        print("Something wrong during coordinate ascent")
                        print(zetad)
                        print(phid)
                
                # If parameters didn't change much, we're done.
                zetachange = n.mean(n.absolute(zetad - lastzetad+1e-100))
                try:
                    phichange = n.mean(n.absolute(phid - lastphid+1e-100))
                except:
                    phichange = 0
                    print("Encountered error at document")
                    print(ids)
                    print(cts)
                gammachange = n.mean(n.absolute(gamma1d - lastgamma1d)) + n.mean(n.absolute(gamma2d - lastgamma2d))
                meanchange = 0.25*(zetachange + phichange) + 0.5*gammachange
                
                if (meanchange < meanchangethresh):
                    converged = True
                    break
            
            # Store topic-to-topic and stick-breaking
            zeta[d,:,:] = zetad
            gamma1[d,:] = gamma1d
            gamma2[d,:] = gamma2d
            
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats["lambda"][:, ids] += n.multiply(n.dot(zetad.transpose(), phid), cts) + 1e-100
            sstats["a"] += (n.sum(zetad,axis=0)).flatten()

        return (zeta, gamma1, gamma2), sstats, ELBO

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
        varparams, _, _ = self.do_e_step([wordobs_ids],[wordobs_cts]) 
        zeta = varparams[0][0] # shape (self._T, self._K)
        gamma1 = varparams[1][0] # shape (self._T,)
        gamma2 = varparams[2][0] # shape (self._T,)
        
        sndlevel = GEM_expectation(gamma1[n.newaxis,:], gamma2[n.newaxis,:]) # shape (1, self._T)
        thetad = n.dot(sndlevel, zeta) # shape (1, self._K)
        
        return thetad

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
    
class T_dSB_DP(_TopicModel):
    """
    Inherit _TopicModel to train truncated HDP. 
    """

    def __init__(self, vocab, K, T, topicfile, D, omega, alpha, eta, tau0, kappa):
        """
        Arguments:
            K: Number of topics i.e. corpus-level truncation
            T: Max number of topics manifested by a document i.e. document-level truncation
            vocab: A set of words to recognize. When analyzing documents, any word
               not in this set will be ignored.
            D: Total number of documents in the population. 
            topicfile: Dictionary containing path to some other topic model's output topics and
                the name of the topic model.
            omega: corpus-level stick-breaking Beta(1,omega)
            alpha: per-document stick-breaking Beta(1,alpha)
            eta: Hyperparameter for prior on topics beta
            tau0: A (positive) learning parameter that downweights early iterations
            kappa: Learning rate: exponential decay rate---should be between
                 (0.5, 1.0] to guarantee asymptotic convergence.
        Remarks:
            The inheritted classes T_dSB_DP and NNFA_HDP might use more 
            instance variables. 
            User should make sure the topics from topicpath are compatible with 
            K and vocab.
        """
        t0 = time.time()
        
        # Common instance variables
        _TopicModel.__init__(self, vocab, K, T, topicfile, D, omega, alpha, eta, tau0, kappa)
        
        # T_dSB_DP specific instance ariables
        if (topicfile is None):
            ## Initialize q(beta|lambda) if topicpath is not None
            self._lambda = np.random.gamma(1.0, 1.0, (self._K, self._W)) \
                                        * self._D*100/(self._K*self._W) + self._eta
            self._a = n.ones(self._K)
            self._ainc = n.zeros(self._K)
            self._b = self._omega*n.ones(self._K)
            self._b[-1] = 0
        else:
            self._lambda, self._a, self._b = self.convert_topics(topicfile)
            # print(self._a)
            # print(self._b)
            self._ainc = self._a - 1.0
            print("Successfully loaded topics from %s" %topicfile["lambda"])
            
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._Elogtheta = expect_log_sticks(self._a, self._b) # shape (self._K,)
        
        # expected topic proportions after initialization/loading pre-trained
        self._Etheta = GEM_expectation(self._a[np.newaxis,:],self._b[np.newaxis,:]).flatten()
        # print(self._Etheta)
        
        t1 = time.time()
        print("Time to initialize %d-topic model, each document using %d topics, is %.2f" %(self._K, self._T, t1-t0))
        return
    
    def convert_topics(self, topicfile):
        """
        Convert pre-trained topics of another method 
        into the suitable format for warm-start training.      
        
        Inputs:
            topicfile = dictionary, topicfile["lambda"] is path to topics
                topicfile["a"] is path to a, topicfile["method"] = "N_dSB_DP"
                for instance. 
        Outputs:
        
        Remarks:
            the choice of a and b for topicfile["method"] = "LDA" feels 
            reasonable, but could be improved.
        """
        topics = n.loadtxt(topicfile["lambda"])
        assert topics.shape == (self._K,self._W), "Wrong shape of topics"
        
        if (topicfile["method"] == "T_dSB_DP"):
            a = n.loadtxt(topicfile["a"])
            b = n.loadtxt(topicfile["b"])
            
        elif (topicfile["method"] == "N_dSB_DP"):
            # re-order topics by sum of Dirichlet parameters
            topicssum = n.sum(topics, axis=1)
            idx = [i for i in reversed(n.argsort(topicssum))]
            topics = topics[idx,:]
            a = n.loadtxt(topicfile["a"])
            a = a[idx]
            a[-1] = 1.0
            ainc = a - 1.0
            # N_dsB_DP doesn't use b, so we have to form new estimates
            # based on a
            b = self._omega + n.dot(self._Kmask, ainc)
            b[-1] = 0
            
        elif (topicfile["method"] == "LDA"):
            # re-order topics by sum of Dirichlet parameters
            topicssum = n.sum(topics, axis=1)
            idx = [i for i in reversed(n.argsort(topicssum))]
            topics = topics[idx,:]
            # compute mean of each topics' top 100 occuring words
            wordsorted = n.fliplr(n.sort(topics,axis=1))
            # print(wordsorted)
            wordmeans = n.mean(wordsorted[:,0:100], axis=1)
            a = n.maximum(wordmeans, 1.0)
            a[-1] = 1.0
            ainc = a - 1.0
            b = self._omega + n.dot(self._Kmask, ainc)
            b[-1] = 0 
        
        return (topics, a, b)
    
    def save_topics(self, savedir, iteration):
        """Save topics"""
        lambdaname = (savedir + "/lambda-%d.dat") % iteration
        n.savetxt(lambdaname, self._lambda)
        aname = (savedir + "/a-%d.dat") % iteration 
        n.savetxt(aname, self._a)
        bname = (savedir + "/b-%d.dat") % iteration 
        n.savetxt(bname, self._b)
        return 
    
    def do_m_step(self, wordids, wordcts, reorder=True):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        topics lambda and the corpus-level stick-breaking a and b.

        Arguments:
            wordids: list of lists
            wordcts: list of lists
            reorder: whether or not to reorder topics based on prevalence
    
        Returns:
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        
        # Do an E step to update zeta, phi, gamma | lambda, a, b for this
        # mini-batch. This also returns the information about phi, zeta that
        # we need to update lambda, a, b
        varparams, sstats, ELBO = self.do_e_step(wordids, wordcts)
        
        # Update global parameters
        
        ## underflow encountered in true_divide after first update
        """
        print("underflow encountered in true_divide at update %d" %self._updatect)
        print(sstats["lambda"])
        print()
        print(n.amax(sstats["lambda"],axis=1))
        print()
        print(sstats["lambda"]*self._D)
        print()
        print(self._lambda)
        print()
        print(rhot)
        print(self._eta)
        print(self._D)
        print(len(wordids))
        """
        
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats["lambda"] / len(wordids))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        
        # self._ainc is equal to self.m_varphi_ss in Chong Wang's code.
        self._ainc = self._ainc * (1-rhot) + \
                rhot * sstats["a"] * self._D / len(wordids)
        
        # reorder topics (helps avoid bad local maxima)
        if (reorder):
            lambdasum = n.sum(self._lambda, axis=1)
            idx = [i for i in reversed(n.argsort(lambdasum))]
            self._lambda = self._lambda[idx,:]
            self._Elogbeta = self._Elogbeta[idx,:]
            self._expElogbeta = self._expElogbeta[idx,:]
            self._ainc = self._ainc[idx]
        
        self._a = 1.0 + self._ainc
        self._a[-1] = 1
        self._b = self._omega + n.dot(self._Kmask, self._ainc)
        self._b[-1] = 0
        self._Elogtheta = expect_log_sticks(self._a, self._b) # shape (self._K,)
        self._Etheta = GEM_expectation(self._a[np.newaxis,:],self._b[np.newaxis,:]).flatten()
        
        self._updatect += 1

        return varparams

class N_dSB_DP(_TopicModel):
    """
    Inherit _TopicModel to train Non-Nested Finite Approximation to HDP. 
    """

    def __init__(self, vocab, K, T, topicfile, D, omega, alpha, eta, tau0, kappa):
        """
        Arguments:
            K: Number of topics i.e. corpus-level truncation
            T: Max number of topics manifested by a document i.e. document-level truncation
            vocab: A set of words to recognize. When analyzing documents, any word
               not in this set will be ignored.
            D: Total number of documents in the population. 
            topicfile: Path to some pre-trained topics' i.e. variational Dirichlet parameters. 
            omega: corpus-level stick-breaking Beta(1,omega)
            alpha: per-document stick-breaking Beta(1,alpha)
            eta: Hyperparameter for prior on topics beta
            tau0: A (positive) learning parameter that downweights early iterations
            kappa: Learning rate: exponential decay rate---should be between
                 (0.5, 1.0] to guarantee asymptotic convergence.
        """
        t0 = time.time()
        
        # Common instance variables
        _TopicModel.__init__(self, vocab, K, T, topicfile, D, omega, alpha, eta, tau0, kappa)
        
        # N_dSB_DP specific instance variables
        ## Initialize q(beta|lambda) if topicfile is not None
        if (topicfile is None):
            self._lambda = np.random.gamma(1.0, 1.0, (self._K, self._W)) \
                            * self._D*100/(self._K*self._W) + self._eta
            self._a = self._omega*n.ones(self._K)/self._K
        else:
            self._lambda, self._a = self.convert_topics(topicfile)
            print("Successfully loaded topics from %s" %topicfile["lambda"])
        
        self._Elogtheta = dirichlet_expectation(self._a) # shape (self._K,)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        
        ## Expected topic proportions
        self._Etheta = self._a/n.sum(self._a)
        
        t1 = time.time()
        print("Time to initialize %d-topic model, each document using %d topics, is %.2f" %(self._K, self._T, t1-t0))
        return
    
    def convert_topics(self, topicfile):
        """
        Convert pre-trained topics of another method 
        into the suitable format for warm-start training.      
        
        Inputs:
            topicfile = dictionary, topicfile["lambda"] is path to topics
                topicfile["a"] is path to a, topicfile["method"] = "N_dSB_DP"
                for instance. 
        Outputs:
        
        Remarks:
            the choice of a and b for topicfile["method"] = "LDA" feels 
            reasonable, but could be improved.
        """
        topics = n.loadtxt(topicfile["lambda"])
        assert topics.shape == (self._K,self._W), "Wrong shape of topics"
        
        if (topicfile["method"] == "N_dSB_DP" or topicfile["method"] == "T_dSB_DP"):
            a = n.loadtxt(topicfile["a"])
            
        elif (topicfile["method"] == "LDA"):
            # compute mean of each topics' top 100 occuring words
            wordsorted = n.sort(topics,axis=1)
            wordmeans = n.mean(wordsorted[:,0:100], axis=1)
            a = wordmeans
        
        return (topics, a)
    
    def save_topics(self, savedir, iteration):
        """Save topics"""
        lambdaname = (savedir + "/lambda-%d.dat") % iteration
        n.savetxt(lambdaname, self._lambda)
        aname = (savedir + "/a-%d.dat") % iteration 
        n.savetxt(aname, self._a)
        return 
    
    def do_m_step(self, wordids, wordcts, reorder=True):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        topics lambda and the corpus-level stick-breaking a and b.

        Arguments:
            wordids: list of lists
            wordcts: list of lists
            reorder: whether or not to reorder topics based on prevalence
    
        Returns:
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        
        # Do an E step to update zeta, phi, gamma | lambda, a, b for this
        # mini-batch. This also returns the information about phi, zeta that
        # we need to update lambda, a, b
        varparams, sstats, ELBO = self.do_e_step(wordids, wordcts)
        
        # Update global parameters
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats["lambda"] / len(wordids))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        
        self._a = (1-rhot)*self._a + \
            rhot * (self._omega/self._K + self._D * sstats["a"] / len(wordids))
        self._Elogtheta = dirichlet_expectation(self._a) # shape (self._K,)
        self._Etheta = self._a/n.sum(self._a)
                    
        self._updatect += 1

        return varparams
    
def sanity_E_step(seed, K, T, topicfile):
    """
    Examine effect of THDP E-step's on a document and the resulting M-step 
    on T_dSB_DP.
    Inputs:
        seed: seed for replicability
        K: cap on corpus-level number of topics
        T: cap on per-document number of topics
        topicpath: file path of pre-trained topics for warm-start training
    Outputs:
    """
    ## load topics
    inroot = "wiki10k"
    infile = inroot + "_wordids.csv"
    with open(infile) as f:
        D = sum(1 for line in f)
    vocab = open('./dictnostops.txt').readlines()
    tm = T_dSB_DP(vocab, K, T, topicfile, D, 1, 1, 0.01, 1024., 0.7)
    
    ## E-step on a document, plotting initial guess of topic proportions 
    ## as well as their convergence, but make no updates to underlying topics 
    n.random.seed(seed)
    (wordids, wordcts) = \
            get_batch_from_disk(inroot, D, 6)
    maxnum = 20
    s = bag_of_words(wordids[0], wordcts[0], tm._idxtoword, maxnum)
    print(s)
    _, _, ELBO = tm.do_e_step(wordids, wordcts, debug=True)
    
    fig1, axes1 = plt.subplots(2,3,figsize=(12,8))
    for i in range(6):
        cidx = i % 3
        ridx = i // 3
        # ELBO
        axes1[ridx, cidx].plot(ELBO[i])
        axes1[ridx, cidx].set_xlabel("Iteration")
        axes1[ridx, cidx].set_ylabel("ELBO")
        axes1[ridx, cidx].set_title("Document %i contains %d unique words" %(i, sum(wordcts[i])))
    fig1.tight_layout()
    plt.show()
    
    return

def sanity_atomic(seed=0, K=100, T=20, topicpath=None, evalLL=False):
    """
    Atomic routine (sample 1 batch, M-step, evaluate held-out LL).
    Mainly for profiling purposes.
    Inputs:
        seed: seed for replicability
        K: cap on corpus-level number of topics
        T: cap on per-document number of topics
        topicpath: file path of pre-trained topics for warm-start training
        evalLL: whether held-out LL should be evaluated
    Outputs:
    """
    inroot = "wiki10k"
    infile = inroot + "_wordids.csv"
    heldoutroot = "wiki1k"
    heldoutfile =  heldoutroot + "_wordids.csv"
    
    with open(infile) as f:
        D = sum(1 for line in f)
    with open(heldoutfile) as f:
        D_ = sum(1 for line in f)
        
    # load the held-out documents
    (howordids,howordcts) = \
                    get_batch_from_disk(heldoutroot, D_, None)
    
    vocab = open('./dictnostops.txt').readlines()
    tm = T_dSB_DP(vocab, K, T, topicpath, D, 1, 1, 0.01, 1024., 0.7)
    
    # Update topics
    n.random.seed(seed)
    (wordids, wordcts) = \
            get_batch_from_disk(inroot, D, 100)
    tm.do_m_step(wordids, wordcts)
    
    # Evaluate held-out LL
    if (evalLL):
        LL = tm.log_likelihood_docs(howordids,howordcts)
    return

def main(evalLL):
    # sanity_E_step(3, 100, 10, None)
    sanity_atomic(evalLL=evalLL)
    return

if __name__ == '__main__':
    main(True)