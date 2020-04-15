import os

import re
import string
import time
import csv, math
import numpy as np
from tqdm import tqdm

# read and organize data

#3 2:3 4:5 5:3 --- document info (word: count)
class document:
    ''' the class for a single document '''
    def __init__(self):
        self.words = []
        self.counts = []
        self.length = 0
        self.total = 0

class corpus:
    ''' the class for the whole corpus'''
    def __init__(self):
        self.size_vocab = 0
        self.docs = []
        self.num_docs = 0

    def read_data(self, filename):
        if not os.path.exists(filename):
            print('no data file, please check it')
            return
        print('reading data from %s.' % filename)

        for line in file(filename): 
            ss = line.strip().split()
            if len(ss) == 0: continue
            doc = document()
            doc.length = int(ss[0])

            doc.words = [0 for w in range(doc.length)]
            doc.counts = [0 for w in range(doc.length)]
            for w, pair in enumerate(re.finditer(r"(\d+):(\d+)", line)):
                doc.words[w] = int(pair.group(1))
                doc.counts[w] = int(pair.group(2))

            doc.total = sum(doc.counts) 
            self.docs.append(doc)

            if doc.length > 0:
                max_word = max(doc.words)
                if max_word >= self.size_vocab:
                    self.size_vocab = max_word + 1

            if (len(self.docs) >= 10000):
                break
        self.num_docs = len(self.docs)
        print("finished reading %d docs." % self.num_docs)

# def read_data(filename):
#     c = corpus()
#     c.read_data(filename)
#     return c

def read_stream_data(f, num_docs):
  c = corpus()
  splitexp = re.compile(r'[ :]')
  for i in range(num_docs):
    line = f.readline()
    line = line.strip()
    if len(line) == 0:
      break
    d = document()
    splitline = [int(i) for i in splitexp.split(line)]
    wordids = splitline[1::2]
    wordcts = splitline[2::2]
    d.words = wordids
    d.counts = wordcts
    d.total = sum(d.counts)
    d.length = len(d.words)
    c.docs.append(d)

  c.num_docs = len(c.docs)
  return c

# This version is about 33% faster
def read_data(filename):
    c = corpus()
    splitexp = re.compile(r'[ :]')
    for line in open(filename):
        d = document()
        splitline = [int(i) for i in splitexp.split(line)]
        wordids = splitline[1::2]
        wordcts = splitline[2::2]
        d.words = wordids
        d.counts = wordcts
        d.total = sum(d.counts)
        d.length = len(d.words)
        c.docs.append(d)

        if d.length > 0:
            max_word = max(d.words)
            if max_word >= c.size_vocab:
                c.size_vocab = max_word + 1

    c.num_docs = len(c.docs)
    return c

def count_tokens(filename):
    num_tokens = 0
    splitexp = re.compile(r'[ :]')
    for line in open(filename):
        splitline = [int(i) for i in splitexp.split(line)]
        wordcts = splitline[2::2]
        num_tokens += sum(wordcts)

    return num_tokens

splitexp = re.compile(r'[ :]')
def parse_line(line):
    line = line.strip()
    d = document()
    splitline = [int(i) for i in splitexp.split(line)]
    wordids = splitline[1::2]
    wordcts = splitline[2::2]
    d.words = wordids
    d.counts = wordcts
    d.total = sum(d.counts)
    d.length = len(d.words)
    return d

def switch_format(root):
    """
    Input:
        root: str, for instance "wiki10k"
    Output:
        join the wordids and wordcts file into one single .txt 
        in the "0 234 10 345 5 ...
                1 684 57 452 3 ... 
                ... " format.
    """
    infile = root + "_wordids.csv"
    with open(infile) as f:
        D = sum(1 for line in f)
        print("Started with %d lines" %D)
    fids = list(csv.reader(open(root + "_wordids.csv")))
    fcts = list(csv.reader(open(root + "_wordcts.csv")))
    outname = root + ".txt"
    outfile = open(outname,'w+')
    for i in tqdm(range(D),desc='Processing line by line'):
        ids = fids[i]
        cts = fcts[i]
        new = ids + cts
        new[0::2] = ids
        new[1::2] = cts # interleave the entries of ids and cts
        new = [str(i)] + new # number the document
        nextline = ' '.join(new)
        outfile.write(nextline + '\n')
    # how many lines were written?
    outfile.close()
    with open(outname) as f:
        Dwritten = sum(1 for line in f)
        print("Wrote %d lines" %Dwritten)
    return 

def parse_doc_list(docs, vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments: 
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns a pair of lists of lists. 

    The first, wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)

    The second, wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    wordids = list()
    wordcts = list()
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(list(ddict.keys()))
        wordcts.append(list(ddict.values()))

    return((wordids, wordcts))

def make_vocab(vocab):
    """
    Inputs:
        vocab: dictionary mapping words to unique integer ids
    Outputs:
        vocabdict: slight modification of vocab, where all words are 
        reduced to their lower case form.
        idxtoword: dictionary mapping integer ids to dictionar words
    """
    vocabdict = dict()
    for word in vocab:
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        vocabdict[word] = len(vocabdict)
    idxtoword = dict()
    for item in vocabdict.items():
        idx, word = item[1], item[0]
        idxtoword[idx] = word
    # print("Vocabdict length %d while idxtoword length %d" %(len(vocabdict),len(idxtoword)))
    return (vocabdict, idxtoword)

def clean_corpus(inroot):
    """
    Input: "wiki10k", for instance
    Output: remove empty lines from corpus
    """
    cleanids = []
    with open(inroot + "_wordids.csv") as f:
        all_lines = list(csv.reader(f))
        for i in range(len(all_lines)):
            if len(all_lines[i]) == 0:
                print("Empty line in ids detected at line %d" %i)
            else:
                cleanids.append(all_lines[i])
                
    cleancts = []
    with open(inroot + "_wordcts.csv") as f:
        all_lines = list(csv.reader(f))
        for i in range(len(all_lines)):
            if len(all_lines[i]) == 0:
                print("Empty line in cts detected at line %d" %i)
            else:
                cleancts.append(all_lines[i])
                
    # Save corpus with no empty lines
    idsname = "clean" + inroot + "_wordids.csv"
    ctsname = "clean" + inroot + "_wordcts.csv"
    with open(idsname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cleanids)
    with open(ctsname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cleancts)
    return

def get_batch_from_disk(inroot, D, batch_size=None):
    """
    Inputs:
        inroot = str, "wiki10k" for example
        D = int, number of documents in training corpus
        batch_size = int, number of documents to sample
    Outputs:
    Remarks: 
        The 10k data-set is small enough (4MB) to be loaded in its 
        entirety. Might need to take care when moving to larger
        data-sets.  
    """

    t0 = time.time()
   
    if (not batch_size is None):
         # generate random indices
        indices = list(np.random.randint(0, D, batch_size))
        load_size = batch_size
    else:
        # load all examples
        indices = list(range(0,D))
        load_size = D

    with open(inroot + "_wordids.csv") as f:
        all_lines = list(csv.reader(f))
        wordids = [list(map(int, all_lines[i])) for i in indices]
    with open(inroot + "_wordcts.csv") as f:
        all_lines = list(csv.reader(f))
        wordcts = [list(map(int, all_lines[i])) for i in indices]

    t1 = time.time()
    # print("Time to taken to get batch of size %d from a total of %d documents is %.2f" %(load_size, D, t1-t0))

    return (wordids, wordcts)

def split_document(docids, doccts, ratio=0.75):
    """
    Split a document into an observed and a held-out part, enforcing
    that the set of unique words in each part are disjoint. Don't 
    split if the document has only one word!
    Inputs:
        docids = list of ints
        doccts = list of ints
        ratio = scalar, by default 0.75
    Outputs: tuple of four lists
        wordobs_ids
        wordobs_cts
        wordho_ids
        wordho_cts
    """
    assert(len(docids)>1), "Tried to split a one-word document"
    Mobs = int(math.floor(len(docids)*ratio))
    obsind = list(range(0, Mobs))
    hoind = list(range(Mobs,len(docids)))
    wordobs_ids = [docids[i] for i in obsind]
    wordobs_cts = [doccts[i] for i in obsind]
    wordho_ids = [docids[i] for i in hoind]
    wordho_cts = [doccts[i] for i in hoind]
    return (wordobs_ids, wordobs_cts, wordho_ids, wordho_cts)

def bag_of_words(docids, doccts, idxtoword, maxnum):
    """
    Convert the docids and doccts representation of a document
    into the more interpretable bag-of-word presentation.
    Inputs:
        docids = list of unique word ids
        doccts = list of how many words occurred
        idxtoword = dictionary mapping word id to word
        maxnum = maximum number of words to print
    Outputs:
        string representing the bag-of-word representation of 
        the document
    """
    s = "Document:\n"
    for i in range(0, min(maxnum,len(docids))):
        word = idxtoword[docids[i]]
        count = doccts[i]
        ws = '%20s  \t---\t  %d \n' %(word,count)
        s = s + ws
    s = s + "\n"
    return s

def main():
    # test bag_of_words
    # np.random.seed(1)
    inroot = "wiki10k"
    infile = inroot + "_wordids.csv"
    with open(infile) as f:
        D = sum(1 for line in f)
    batchsize = 5
    (wordids, wordcts) = \
            get_batch_from_disk(inroot, D, batchsize)
    vocab = open('./dictnostops.txt').readlines()
    _, idxtoword = make_vocab(vocab)
    maxnum = 20
    for i in range(batchsize):
        s = bag_of_words(wordids[i], wordcts[i], idxtoword, maxnum)
        print(s)
    return

if __name__ == '__main__':
    main()
