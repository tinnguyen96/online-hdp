import pickle, string, numpy, getopt, sys, random, time, re, pprint
import os
import argparse
import matplotlib.pyplot as plt

import topicmodelvb
import corpus
from utils import TrainSpecs

def makesaves(K, T, batchsize, inroot, heldoutroot, seed, topicfile, method, tau, kappa):
    """ Make save paths (and directories, if necessary) for experiment """
    
    savedir = "results/" + method + "K" + str(K) + "_T" + str(T) + "_D" + str(batchsize) + "_" + inroot + "_" + heldoutroot
    savedir = (savedir + "/tau%.1f_kappa%.1f") % (tau, kappa)
    savedir = savedir.replace(".",",")
    if (not topicfile is None):
        topic = topicfile["lambda"].replace(".dat","")
        topic = topic.replace("results/","")
        savedir = savedir + "/warm/" + topic
    LLsavename = savedir + "/LL_" + str(seed) + ".csv"
    if not os.path.exists(savedir):
        os.makedirs(savedir,0o777,True)
        print("Succesfully created directory %s" %savedir)
    return savedir, LLsavename 

def maketopicfile(topicinfo):
    """Create dictionary that topic models need to load pre-trained topics """
    
    if (topicinfo is None):
        topicfile = None
    else:
        othermethod = topicinfo[0]
        iteration = topicinfo[2]
        topicfile = {}
        topicfile["method"] = othermethod
        topicfile["lambda"] = topicinfo[1] + "lambda-" + iteration + ".dat"
        if (othermethod == "N_dSB_DP"):
            topicfile["a"] = topicinfo[1] + "a-" + iteration + ".dat"
        elif (othermethod == "T_dSB_DP"):
            topicfile["a"] = topicinfo[1] + "a-" + iteration + ".dat"
            topicfile["b"] = topicinfo[1] + "b-" + iteration + ".dat"
    return topicfile

def run_experiment(args):
    
    # The rootname, for instance wiki10k
    inroot = args.inroot
    infile = inroot + "_wordids.csv"

    # For instance, wiki1k
    heldoutroot = args.heldoutroot 
    heldoutfile = args.heldoutroot + "_wordids.csv"
    
    with open(infile) as f:
        D = sum(1 for line in f)
    print(("Training corpus has %d documents" %D))

    with open(heldoutfile) as f:
        D_ = sum(1 for line in f)
    print(("Held-out corpus has %d documents" %D_))

    # Set random seed for replicability. Random sampling of 
    # mini-batches.
    seed = args.seed
    numpy.random.seed(seed)

    # The number of documents to analyze each iteration
    batchsize = args.batchsize

    # Total number of batches
    if args.maxiter is None:
        max_iter = 1000
    else:
        max_iter = args.maxiter
        
    # forgetting rate 
    kappa = args.kappa
    
    # downplay early iterations
    tau = args.tau
    
    # Spacing between computing average train time
    progressiter = args.progressiter
    
    # Spacing between saving topic 
    topiciter = args.topiciter
    
    # Spacing between evaluating held-out log-likelihood
    LLiter = args.LLiter
    
    print("Save topics every %d iterations, compute average train time every %d iterations, report likelihood every %d iterations" %(topiciter, progressiter, LLiter))
    
    dotrain = (args.train == "True")
    dotest = (args.test == "True")
    
    print("Do we train? %s. Do we test? %s" %(dotrain, dotest))
    
    # Our vocabulary
    vocab = open('./dictnostops.txt').readlines()
    W = len(vocab)
    
    # Paths for warm-start training
    topicfile = maketopicfile(args.topicinfo)
    
     # load the held-out documents
    (howordids,howordcts) = \
                    corpus.get_batch_from_disk(heldoutroot, D_, None)

    # experiments for different number of topics
    Klist = args.K
    T = args.T
    for K in Klist:
        LL_list = []
        # Different constructors for different methods
        method = args.method
        if (method == "nhdp"):
            tm = topicmodelvb.N_dSB_DP(vocab, K, T, topicfile, D, 1, 1, 0.01, tau, kappa)
        elif (method == "thdp"):
            tm = topicmodelvb.T_dSB_DP(vocab, K, T, topicfile, D, 1, 1, 0.01, tau, kappa)
        elif (method == "lda"):
            tm = topicmodelvb.LDA(vocab, K, topicfile, D, 1, 0.01, tau, kappa)
            
        train_time = 0
        savedir, LLsavename = makesaves(K, T, batchsize, inroot, heldoutroot, seed, topicfile, method, tau, kappa)
       
        if (not topicfile is None):
            ## Are the pre-trained topics a good initialization?
            
            if (method == "nhdp" or method == "thdp"):
                ## Save plot of expected proportions
                Etheta = tm._Etheta
                plt.figure()
                plt.plot(Etheta, marker='o', markersize=4, label='Expected proportions',color='b')
                plt.title("Expected topic proportions at initialization")
                plt.xlabel("Topic index")
                plt.ylabel("Expected proportions")
                plt.savefig(savedir + "/expected_proportions_@init.png")
            
            ## Plot initial held-out log-likelihood
            initLL = tm.log_likelihood_docs(howordids,howordcts)
            print("Under warm-start topics, current model has held-out LL: %f" %initLL)
        
        ## Do we train or just call it a day ...
        if (dotrain):
            print("Started training ...")
            for iteration in range(0, max_iter):
                t0 = time.time()
                # Load a random batch of articles from disk
                (wordids, wordcts) = \
                    corpus.get_batch_from_disk(inroot, D, batchsize)
                # Give them to topic model
                tm.do_m_step(wordids, wordcts)
                t1 = time.time()
                train_time += t1 - t0

                if (iteration % progressiter == 0):
                    print('seed %d, iter %d:  rho_t = %f,  cumulative train time = %f,  average train time = %.2f' % \
                        (seed, iteration, tm._rhot, train_time, train_time/(iteration+1)))

                # should we report held-out LL
                if (dotest):
                    # Compute average log-likelihood on held-out corpus every so number of iterations
                    if (iteration % LLiter == 5):
                        t0 = time.time()
                        LL = tm.log_likelihood_docs(howordids,howordcts)
                        t1 = time.time()
                        test_time = t1 - t0
                        print('\t\t seed %d, iter %d:  rho_t = %f,  test time = %f, held-out log-likelihood = %f' % \
                            (seed, iteration, tm._rhot, test_time, LL))
                        LL_list.append([iteration, train_time, LL])
                        numpy.savetxt(LLsavename, LL_list)

                # save topics every so number of iterations
                if (seed == 0):
                    if (iteration % topiciter == 0):
                        tm.save_topics(savedir, iteration)

            print("Finished experiment with top level %d, second level %d,  %s model" %(K, T, method))
            
    return 
    
def main():
    """
    Read command-line arguments to train a topic model using wikipedia corpora.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="whether to train or just load pre-trained topics and report LL [True]", default="True")
    parser.add_argument("--test", help="whether to report LL on held-out documents [True]", default="True")
    parser.add_argument("--method", help="type of topic model [thdp, nhdp]", default='thdp')
    parser.add_argument("--K", help="cap of corpus-level number of topics, expecting a list [[100]]",nargs='+',type=int, default=[100])
    parser.add_argument("--T", help="cap of document-level number of topics [10]",type=int, default=10)
    parser.add_argument("--tau", help="downplay role of early iterations [1024]",type=float, default=1024.0)
    parser.add_argument("--kappa", help="forgetting rate [0.7]",type=float, default=0.7)
    parser.add_argument("--LLiter", help="number of iterations between evaluating held-out log-likelihood [100]",type=int, default=100)
    parser.add_argument("--progressiter", help="number of iterations between reporting average train time [10]", type=int, default=10)
    parser.add_argument("--topiciter", help="number of iterations between saving topics [100]",type=int, default=100)
    parser.add_argument("--inroot", help="training corpus root name [wiki10k]", default='wiki10k')
    parser.add_argument("--heldoutroot", help="testing corpus root name [wiki1k]", default='wiki1k')
    parser.add_argument("--topicinfo",help="information on pre-trained topics for warm-start. Don't set if numtopics is a list, [['LDA','results/lda_K100_D50_wiki10k_wiki1k/', '100']]",nargs='+', default=None)
    parser.add_argument("--seed", help="seed for replicability [0]",type=int, default=0)
    parser.add_argument("--maxiter", help="total number of mini-batches to train [1000]",type=int, default=1000)
    parser.add_argument("--batchsize", help="mini-batch size [20]",type=int, default=20)
    args = parser.parse_args()
    numpy.seterr(all='raise')
    run_experiment(args)
    return

def evaltopics(predictivemodel, topicroot, iterations, ax):
    """
    Does training for longer mean the topics serve as better warm-start
    for other models?
     
    Inputs:
        predictivemodel: dictionary, predictivemodel["method"] = "thdp",
            predictivemodel["K"] = 100, ...
        topicroot: "thdp_K100_T10_D20_wiki10k_wiki1k", for instance
        iterations: "[100, 200, 300]", for instance
        ax: where to plot
    Outputs:
        log-likelihood of predictive model using the topics from topicroot
        at iterations
    """
    for t in iterations:
        # load predictive model
        if (predictivemodel["method"] == "thdp"):
            tm = topicmodelvb.T_dSB_DP(vocab, K, T, topicfile, D, 1, 1, 0.01, tau, kappa)
        elif (predictivemodel["method"] == "nhdp"):
            tm = topicmodelvb.N_dSB_DP(vocab, K, T, topicfile, D, 1, 1, 0.01, tau, kappa)
    print()
        

    return 

def debug(method, topicinfo):
    """
    Test for correctness of various things by passing in arguments of interest
    without the command-line.
    """
    # Warm-start training
    args = TrainSpecs(train="False", test="True", method=method, K=[100], T=10, LLiter=100, progressiter=10, topiciter=100,
                      inroot='wiki10k', heldoutroot='wiki1k', 
                      topicinfo=topicinfo, seed=0, 
                      maxiter=1000, batchsize=20)
    run_experiment(args)
        
if __name__ == '__main__':
    main()
