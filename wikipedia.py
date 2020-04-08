import pickle, string, numpy, getopt, sys, random, time, re, pprint
import os
import argparse

import topicmodelvb
import corpus

def makesaves(K, T, batchsize, inroot, heldoutroot, seed, topicpath, method):
    savedir = "results/" + method + "K" + str(K) + "_T" + str(T) + "_D" + str(batchsize) + "_" + inroot + "_" + heldoutroot
    if (not topicpath is None):
        savedir = savedir + "/warm/" + topicpath
    LLsavename = savedir + "/LL_" + str(seed) + ".csv"
    if not os.path.exists(savedir):
        os.makedirs(savedir,0o777,True)
        print("Succesfully created directory %s" %savedir)
    return savedir, LLsavename 

def main():
    """
    Load a wikipedia corpus in batches from disk and run either T-HDP or N-HDP.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="type of topic model [thdp, nhdp]", default='thdp')
    parser.add_argument("--K", help="cap of corpus-level number of topics, expecting a list",nargs='+',type=int, default=[100])
    parser.add_argument("--T", help="cap of document-level number of topics",type=int, default=20)
    parser.add_argument("--LLiter", help="number of iterations between evaluating held-out log-likelihood",type=int, default=100)
    parser.add_argument("--progressiter", help="number of iterations between reporting average train time", type=int, default=10)
    parser.add_argument("--topiciter", help="number of iterations between saving topics",type=int, default=100)
    parser.add_argument("--inroot", help="training corpus root name", default='wiki10k')
    parser.add_argument("--heldoutroot", help="testing corpus root name", default='wiki1k')
    parser.add_argument("--topicpath",help="path to pre-trained topics for warm-start. Don't set if numtopics is a list", default=None)
    parser.add_argument("--seed", help="seed for replicability",type=int, default=0)
    parser.add_argument("--maxiter", help="total number of mini-batches to train",type=int, default=1000)
    parser.add_argument("--batchsize", help="mini-batch size",type=int, default=20)
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

    # Total number of batches
    if args.maxiter is None:
        max_iter = 1000
    else:
        max_iter = args.maxiter
    
    # Spacing between computing average train time
    progressiter = args.progressiter
    
    # Spacing between saving topic 
    topiciter = args.topiciter
    
    # Spacing between evaluating held-out log-likelihood
    LLiter = args.LLiter
    
    print("Save topics every %d iterations and report likelihood every %d iterations" %(topiciter, LLiter))
    
    # Our vocabulary
    vocab = open('./dictnostops.txt').readlines()
    W = len(vocab)
    
    # Whether to do warmstart
    topicpath = args.topicpath
    if (topicpath is None):
        topicfile = None
    else:
        topicfile = topicpath + ".dat"
        
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
            tm = topicmodelvb.N_HDP(vocab, K, T, topicfile, D, 1, 1, 0.01, 1024., 0.7)
        elif (method == "thdp"):
            tm = topicmodelvb.T_HDP(vocab, K, T, topicfile, D, 1, 1, 0.01, 1024., 0.7)
        train_time = 0
        savedir, LLsavename = makesaves(K, T, batchsize, inroot, heldoutroot, seed, topicpath, method)
       
        if (not topicpath is None):
            initLL = tm.log_likelihood_docs(howordids,howordcts)
            print("Under warm start topics, current model has held-out LL: %f" %initLL)
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
            
            # Compute average log-likelihood on held-out corpus every so number of iterations
            if (iteration % LLiter == 0):
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
                    lambdaname = (savedir + "/lambda-%d.dat") % iteration
                    numpy.savetxt(lambdaname, tm._lambda)
                    aname = (savedir + "/a-%d.dat") % iteration 
                    numpy.savetxt(aname, tm._a)
                    if (method == "thdp"):
                        bname = (savedir + "/b-%d.dat") % iteration 
                        numpy.savetxt(bname, tm._b)

        print("Finished experiment with %d-topic %s model" %(K, T, method))
if __name__ == '__main__':
    main()
