import numpy as n
from scipy.special import gammaln, psi, beta

class TrainSpecs:
    def __init__(self, train=False, method='thdp', K=[100], T=10, LLiter=100, progressiter=10, topiciter=100,
                      inroot='wiki10k', heldoutroot='wiki1k', 
                      topicinfo=['LDA','results/lda_K100_D50_wiki10k_wiki1k/','100'], seed=0, 
                      maxiter=1000, batchsize=20):
        self.train = train
        self.method = method
        self.K = K
        self.T = T
        self.LLiter = LLiter
        self.progressiter = progressiter
        self.topiciter = topiciter
        self.maxiter = maxiter
        self.seed = seed
        self.inroot = inroot
        self.heldoutroot = heldoutroot
        self.topicinfo = topicinfo
        self.batchsize = batchsize
        return 

def dirichlet_expectation(alpha):
    """
    Inputs:
        alpha: K x V, the Dirichlet parameters are stored in rows.
    Outputs:
        For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def expect_log_sticks(sticks1, sticks2):
    """
    Inputs:
        sticks1: (T,) array
        sticks2: (T,) array. Note that sticks2[-1] = 0
    Outputs:
        E[log pi (V)] where V are the stick-breaking variables and pi(V)
        is the corresponding size-biased representation i.e. 
        pi_i = V_i prod_{j=1}^{i-1} (1-V_j). 
    """
    assert sticks2[-1] == 0, "Wrong input to expect_log_sticks"
    ElogVm1Vd = dirichlet_expectation(n.column_stack((sticks1,sticks2)))
    ElogVd = ElogVm1Vd[:,0] # shape (T,). 
    Elogm1Vd = ElogVm1Vd[:,1] # shape (T,)
    Elogm1Vd[-1] = 0
    cumsums = n.cumsum(Elogm1Vd) # entry i = sum_{j=1}^{i} E[log(1-V_j)]
    Elogthetad = ElogVd + cumsums - Elogm1Vd
    return Elogthetad

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
    cumu = n.cumprod(Em1p, axis=1) # shape (K,)
    ratiop = Ep/Em1p # shape (1, K)
    theta = n.multiply(ratiop, cumu) # shape (1, K)
    return theta

def beta_KL(alpha1, beta1, alpha2, beta2):
    """
    Inputs:
        alpha1, beta1, alpha2, beta2: 1-D arrays of positive reals, same length (or some 
        that is compatible with broadcasting)
    Return KL(Beta(alpha1, beta1)||Beta(alpha2, beta2))
    """ 
    div = n.log(beta(alpha2, beta2)/beta(alpha1, beta1)) + (alpha1 - alpha2)*psi(alpha1)  \
    + (beta1 - beta2)*psi(beta1) + (alpha2 + beta2 - alpha1 - beta1)*psi(alpha1 + beta1)
    return div

def dirichlet_KL(lambdap, lambdaq):
    """
    Inputs:
        lambdap, lambdaq: K x V matrix of parameters whose rows describe two dirichlet distributions
    Outputs:
        KL(Dirichlet(lambdap) || Dirichlet(lambdaq)), shape (K,)
    """
    rowsump = n.sum(lambdap, axis=1) # shape (K,)
    rowsumq = n.sum(lambdaq, axis=1) # shape (K,)
    term1 = gammaln(rowsump) - gammaln(rowsumq) # shape (K,)
    #  - log Gamma(sum_{v} lambdap_{k,v}) + log Gamma(sum_{v} lambdaq_{k,v}) 
    term2 = n.sum(gammaln(lambdaq), axis=1) - n.sum(gammaln(lambdap), axis=1) # shape (K,)
    # psi(lambdap_{k,v}) - psi(sum_{v'} lambdap_{k,v'})
    psirowsump = psi(rowsump)
    diff = psi(lambdap) - psirowsump[:,n.newaxis]
    temp = n.multiply(lambdap-lambdaq, diff)
    term3 = n.sum(temp, axis=1) # shape (K,)
    return term1 + term2 + term3