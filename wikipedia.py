import pickle, string, numpy, getopt, sys, random, time, re, pprint
import os
import argparse

import topicmodelvb
import corpus

def makesaves(K, batchsize, inroot, heldoutroot, seed, topicpath, method, Einit):
    savedir = "results/" + method + "K" + str(K) + "_D" + str(batchsize) + "_Einit" + Einit + "_" + inroot + "_" + heldoutroot
    if (not topicpath is None):
        savedir = savedir + "/warm/" + topicpath
    LLsavename = savedir + "/LL_" + str(seed) + ".csv"
    if not os.path.exists(savedir):
        os.makedirs(savedir,0o777,True)
        print("Succesfully created directory %s" %savedir)
    return savedir, LLsavename 

def main():
    """
    Load a wikipedia corpus in batches from disk and run either LDA or SB-LDA.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="type of topic model")
    parser.add_argument("--inroot", help="training corpus root name")
    parser.add_argument("--heldoutroot", help="testing corpus root name")
    parser.add_argument("--topicpath",help="path to pre-trained topics for warm-start. Don't set if numtopics is a list")
    parser.add_argument("--Einit", help="which set of variational parameters to initialize in E-step")
    parser.add_argument("--seed", help="seed for replicability",type=int)
    parser.add_argument("--maxiter", help="total number of mini-batches to train",type=int)
    parser.add_argument("--batchsize", help="mini-batch size",type=int)
    parser.add_argument("--numtopics", help="maximum number of topics, expecting a list",nargs='+',type=int)
    args = parser.parse_args()
    
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

    # The number of topics
    K = args.numtopics

    # Total number of batches
    if args.maxiter is None:
        max_iter = 1000
    else:
        max_iter = args.maxiter
    
    # Our vocabulary
    vocab = open('./dictnostops.txt').readlines()
    W = len(vocab)
    
    # Whether to do warmstart
    topicpath = args.topicpath
    if (topicpath is None):
        topicfile = None
    else:
        topicfile = topicpath + ".dat"
        
    # Which variational parameters to initialize in E-step
    if (args.Einit is None):
        Einit = "proportions"
    else:
        Einit = args.Einit
        
     # load the held-out documents
    (howordids,howordcts) = \
                    corpus.get_batch_from_disk(heldoutroot, D_, None)

    # experiments for different number of topics
    Klist = args.numtopics
    for K in Klist:
        LL_list = []
        # Different constructors for different methods
        method = args.method
        if (method == "lda"):
            lda = topicmodelvb.LDA(vocab, K, topicfile, Einit, D, 1, 0.01, 1024., 0.7)
        elif (method == "sblda"):
            lda = topicmodelvb.SB_LDA(vocab, K, topicfile, Einit, D, 1, 0.01, 1024., 0.7)
        train_time = 0
        savedir, LLsavename = makesaves(K, batchsize, inroot, heldoutroot, seed, topicpath, method, Einit)
       
        if (not topicpath is None):
            initLL = lda.log_likelihood_docs(howordids,howordcts)
            print("Under warm start topics, current model has held-out LL: %f" %initLL)
        for iteration in range(0, max_iter):
            t0 = time.time()
            # Load a random batch of articles from disk
            (wordids, wordcts) = \
                corpus.get_batch_from_disk(inroot, D, batchsize)
            # Give them to SB_LDA
            _ = lda.update_lambda(wordids, wordcts)
            t1 = time.time()
            train_time += t1 - t0
            # Compute average log-likelihood on held-out corpus every so number of iterations
            if (iteration % 10 == 0):
                t0 = time.time()
                LL = lda.log_likelihood_docs(howordids,howordcts)
                t1 = time.time()
                test_time = t1 - t0
                print('seed %d, iter %d:  rho_t = %f,  cumulative train time = %f,  test time = %f,  held-out log-likelihood = %f' % \
                    (seed, iteration, lda._rhot, train_time, test_time, LL))
                LL_list.append([iteration, train_time, LL])
                numpy.savetxt(LLsavename, LL_list)
            # save topics every so number of iterations
            if (seed == 0):
                if (iteration % 400 == 0):
                    lambdaname = (savedir + "/lambda-%d.dat") % iteration
                    numpy.savetxt(lambdaname, lda._lambda)

        print("Finished experiment with %d-topic model" %K)
if __name__ == '__main__':
    main()
