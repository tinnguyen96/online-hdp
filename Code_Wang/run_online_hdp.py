import sys, os
from corpus import *
import onlinehdp
import pickle
import random, time
from numpy import cumsum, sum

import argparse
from glob import glob
np = onlinehdp.np

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(T=100, K=10, D=9965, vocab="dictnostops.txt", eta=0.01, alpha=1.0, gamma=1.0,
                      kappa=0.7, tau=1024, batchsize=100, max_time=10000,
                      max_iter=1000, var_converge=0.0001, random_seed=0, 
                      corpus_name="wiki", data_path="wiki10k.txt", test_data_path="wiki1k.txt", topic_info = None,
                      directory="results", save_lag=500, scale_save_lag=2, pass_ratio=0.5,
                      new_init=False, scale=1.0, adding_noise=False,
                      seq_mode=False, fixed_lag=False)

    parser.add_argument("--T", type=int, dest="T",
                    help="top level truncation [300]")
    parser.add_argument("--K", type=int, dest="K",
                    help="second level truncation [20]")
    parser.add_argument("--D", type=int, dest="D",
                    help="number of documents [-1]")
    parser.add_argument("--vocab", type=str, dest="vocab",
                    help="vocabulary file [dictnostops.txt]")
    parser.add_argument("--eta", type=float, dest="eta",
                    help="the topic Dirichlet [0.01]")
    parser.add_argument("--alpha", type=float, dest="alpha",
                    help="alpha value [1.0]")
    parser.add_argument("--gamma", type=float, dest="gamma",
                    help="gamma value [1.0]")
    parser.add_argument("--kappa", type=float, dest="kappa",
                    help="learning rate [0.5]")
    parser.add_argument("--tau", type=float, dest="tau",
                    help="slow down [1.0]")
    parser.add_argument("--batchsize", type=int, dest="batchsize",
                    help="batch size [100]")
    parser.add_argument("--max_time", type=int, dest="max_time",
                    help="max time to run training in seconds [100]")
    parser.add_argument("--max_iter", type=int, dest="max_iter",
                    help="max iteration to run training [-1]")
    parser.add_argument("--var_converge", type=float, dest="var_converge",
                    help="relative change on doc lower bound [0.0001]")
    parser.add_argument("--random_seed", type=int, dest="random_seed",
                    help="the random seed [999931111]")
    parser.add_argument("--corpus_name", type=str, dest="corpus_name",
                    help="the corpus name: nature, nyt or wiki [None]")
    parser.add_argument("--data_path", type=str, dest="data_path",
                    help="training data path or pattern [None]")
    parser.add_argument("--test_data_path", type=str, dest="test_data_path",
                    help="testing data path [None]")
    parser.add_argument("--topic_info",help="information on pre-trained topics for warm-start. Don't set if numtopics is a list, [['LDA','results/lda_K100_D50_wiki10k_wiki1k/', '100']]",nargs='+')
    parser.add_argument("--directory", type=str, dest="directory",
                    help="output directory [None]")
    parser.add_argument("--save_lag", type=int, dest="save_lag",
                    help="the minimal saving lag, increasing as save_lag * 2^scale_save_lag, with max i as 10; default 500.")
    parser.add_argument("--scale_save_lag", type=int, dest="scale_save_lag",
                    help="scale factor of saving lag, increasing as save_lag * 2^scale_save_lag; default 1")
    parser.add_argument("--pass_ratio", type=float, dest="pass_ratio",
                    help="The pass ratio for each split of training data [0.5]")
    parser.add_argument("--new_init", action="store_true", dest="new_init",
                    help="use new init or not")
    parser.add_argument("--scale", type=float, dest="scale",
                    help="scaling parameter for learning rate [1.0]")
    parser.add_argument("--adding_noise", action="store_true", dest="adding_noise",
                    help="adding noise to the first couple of iterations or not")
    parser.add_argument("--seq_mode", action="store_true", dest="seq_mode",
                    help="processing the data in the sequential mode")
    parser.add_argument("--fixed_lag", action="store_true", dest="fixed_lag",
                    help="fixing a saving lag")

    options = parser.parse_args()
    return options 

def run_online_hdp():
    # Command line options.
    options = parse_args()

    # Set the random seed.
    random.seed(options.random_seed)
    if options.seq_mode:
        train_file = open(options.data_path)
    else:
        train_filenames = glob(options.data_path)
        train_filenames.sort()
        num_train_splits = len(train_filenames)
        # This is used to determine when we reload some another split.
        num_of_doc_each_split = options.D/num_train_splits 
        # Pick a random split to start
        # cur_chosen_split = int(random.random() * num_train_splits)
        cur_chosen_split = 0 # deterministic choice
        cur_train_filename = train_filenames[cur_chosen_split]
        c_train = read_data(cur_train_filename)
  
    if options.test_data_path is not None:
        test_data_path = options.test_data_path
        c_test = read_data(test_data_path)
        c_test_word_count = sum([doc.total for doc in c_test.docs])
        
    topicfile = maketopicfile(options.topic_info)

    result_directory = "%s/corpus-%s-kappa-%.1f-tau-%.f-batchsize-%d" % (options.directory,
                                                                       options.corpus_name,
                                                                       options.kappa, 
                                                                       options.tau, 
                                                                       options.batchsize)
    if not topicfile is None:
        result_directory = result_directory + "-init-%s%s" %(options.topic_info[0], options.topic_info[2])

    print("creating directory %s" % result_directory)
    if not os.path.isdir(result_directory):
        os.makedirs(result_directory)

    options_file = open("%s/options.dat" % result_directory, "w")
    for opt, value in list(options.__dict__.items()):
        options_file.write(str(opt) + " " + str(value) + "\n")
    options_file.close()

    print("creating online hdp instance.")
    # load vocab file to compute W
    vocab = open(options.vocab).readlines()
    W = len(vocab)
    
    ohdp = onlinehdp.online_hdp(options.T, options.K, options.D, W, topicfile,
                              options.eta, options.alpha, options.gamma,
                              options.kappa, options.tau, options.scale,
                              options.adding_noise)
    if options.new_init:
        ohdp.new_init(c_train)

    print("setting up counters and log files.")

    iter = 0
    save_lag_counter = 0
    total_time = 0.0
    total_doc_count = 0
    split_doc_count = 0
    doc_seen = set()
    log_file = open("%s/log.dat" % result_directory, "w") 
    log_file.write("iteration time doc.count score word.count unseen.score unseen.word.count\n")

    if options.test_data_path is not None:
        test_log_file = open("%s/test-log_seed%d.dat" % (result_directory, options.random_seed), "w") 
        test_log_file.write("iteration time doc.count score word.count avg-score\n")

    print("starting online variational inference.")
    while True:
        iter += 1
        if iter % 100 == 1:
            print("iteration: %05d" % iter)
        t0 = time.clock()

        # Sample the documents.
        batchsize = options.batchsize
        if options.seq_mode:
            c = read_stream_data(train_file, batchsize) 
            batchsize = c.num_docs
            if batchsize == 0:
                break
            docs = c.docs
            unseen_ids = list(range(batchsize))
        else:
            ids = random.sample(list(range(c_train.num_docs)), batchsize)
            docs = [c_train.docs[id] for id in ids]
            # Record the seen docs.
            unseen_ids = set([i for (i, id) in enumerate(ids) if (cur_chosen_split, id) not in doc_seen])
            if len(unseen_ids) != 0:
                doc_seen.update([(cur_chosen_split, id) for id in ids]) 

        total_doc_count += batchsize
        split_doc_count += batchsize

        # Do online inference and evaluate on the fly dataset
        (score, count, unseen_score, unseen_count) = ohdp.process_documents(docs, options.var_converge, unseen_ids)
        total_time += time.clock() - t0
        log_file.write("%d %d %d %.5f %d %.5f %d\n" % (iter, total_time,
                        total_doc_count, score, count, unseen_score, unseen_count))
        log_file.flush()

        # Evaluate on the test data: fixed and folds
        if total_doc_count % options.save_lag == 0:
            if not options.fixed_lag and save_lag_counter < 10:
                save_lag_counter += 1
                options.save_lag = options.save_lag * options.scale_save_lag

            # Save the topics for one random_seed
            if (options.random_seed == 0):
                topicsfile = '%s/doc_count-%d.topics' %  (result_directory, total_doc_count)
                sticksfile = {}
                sticksfile["a"] = '%s/doc_count-%d.a' %  (result_directory, total_doc_count)
                sticksfile["b"] = '%s/doc_count-%d.b' %  (result_directory, total_doc_count)
                ohdp.save_topics(topicsfile, sticksfile)

            if options.test_data_path is not None:
                print("\tconvert hdp to almost equivalent lda.")
                (lda_alpha, lda_beta, lda_Elogbeta) = ohdp.hdp_to_lda()
                print("\tworking on fixed test data.")
                test_score = 0.0
                test_score_split = 0.0
                c_test_word_count_split = 0
                for doc in c_test.docs:
                    (likelihood, count, gamma) = onlinehdp.lda_e_step_split(doc, lda_alpha, lda_Elogbeta, lda_beta)
                    test_score += likelihood
                    c_test_word_count_split += count
                avg_score = test_score/c_test_word_count_split

                test_log_file.write("%d %d %d %.5f %d %.5f \n" % (iter, total_time,
                                    total_doc_count, test_score, c_test_word_count_split, avg_score))
                test_log_file.flush()

        # read another split.
        if not options.seq_mode:
            if split_doc_count > num_of_doc_each_split * options.pass_ratio and num_train_splits > 1:
                print("Loading a new split from the training data")
                split_doc_count = 0
                # cur_chosen_split = int(random.random() * num_train_splits)
                cur_chosen_split = (cur_chosen_split + 1) % num_train_splits
                cur_train_filename = train_filenames[cur_chosen_split]
                c_train = read_data(cur_train_filename)

        if (options.max_iter != -1 and iter > options.max_iter) or (options.max_time !=-1 and total_time > options.max_time):
              break
    log_file.close()
    
    if (options.random_seed == 0):
        print("Saving the final model and topics.")
        topicsfile = '%s/final.topics' %  result_directory
        sticksfile = {}
        sticksfile["a"] = '%s/final.a' %  result_directory
        sticksfile["b"] = '%s/final.b' %  result_directory
        ohdp.save_topics(topicsfile, sticksfile)
    
    if options.seq_mode:
        train_file.close()

    # Makeing final predictions.
    if options.test_data_path is not None:
        (lda_alpha, lda_beta, lda_Elogbeta) = ohdp.hdp_to_lda()
        print("\tworking on fixed test data.")
        test_score = 0.0
        c_test_word_count_split = 0
        for doc in c_test.docs:
            (likelihood, count, gamma) = onlinehdp.lda_e_step_split(doc, lda_alpha, lda_Elogbeta, lda_beta)
            test_score += likelihood
            c_test_word_count_split += count

        test_log_file.write("%d %d %d %.5f %d %.5f \n" % (iter, total_time,
                            total_doc_count, test_score, c_test_word_count_split, avg_score))
        test_log_file.flush()

if __name__ == '__main__':
    run_online_hdp()
