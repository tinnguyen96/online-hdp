import pickle, string, numpy, getopt, sys, random, time, re, pprint
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import topicmodelvb
import corpus
from utils import TrainSpecs
from wikipedia import maketopicfile

vocab = open('./dictnostops.txt').readlines()
W = len(vocab)

def evaltopics(args, topicroot, iterations, ax):
    """
    Does training for longer mean the topics serve as better warm-start
    for other models?
     
    Inputs:
        args: C-struct like object, args.method = "thdp", for instance
        topicroot: ["LDA", "results/lda_K100_D50_wiki10k_wiki1k/"], for instance
        iterations: [100, 200, 300], for instance
        ax: where to plot
    Outputs:
        log-likelihood of predictive model using the topics from topicroot
        at iterations
    """
    heldoutfile = args.heldoutroot + "_wordids.csv"
    with open(heldoutfile) as f:
        D_ = sum(1 for line in f)
    # load the held-out documents
    (howordids,howordcts) = \
                    corpus.get_batch_from_disk(args.heldoutroot, D_, None)
    LL = []
    for t in tqdm(iterations):
        # load predictive model. Some SVI parameters are unimportant for evaluating LL
        # so we set to None.
        topicinfo = topicroot + [str(t)]
        topicfile = maketopicfile(topicinfo)
        if (args.method == "thdp"):
            tm = topicmodelvb.T_dSB_DP(vocab, args.K[0], args.T, topicfile, None, 1, 1, 0.01, args.tau, args.kappa)
        elif (args.method == "nhdp"):
            tm = topicmodelvb.N_dSB_DP(vocab, args.K[0], args.T, topicfile, None, 1, 1, 0.01, args.tau, args.kappa)
        LL.append(tm.log_likelihood_docs(howordids,howordcts))
    print(LL)
    
    ax.plot(LL,marker='o')
    title = args.method + "K" + str(args.K[0]) + "_T" + str(args.T) + "_D" + str(args.batchsize)
    ax.set_title(title)
    ax.set_xlabel("Number of LDA mini-batches")
    ax.set_ylabel("Held-out LL")
    return LL


def main():
    return 

if __name__ == "main":
    main()