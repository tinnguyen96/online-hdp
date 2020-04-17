# Seq_mode

## What is the format of the train set?

A corpus is probably represented as:
"
0 234 10 345 5 ...
1 684 57 452 3 ...
...
"
where the 0th document contains 10 occurences of the word with id 234 ... 
      the 1st document contains 57 occurences of the word with id 684 ... 
      ...

Going from our wiki10k_wordids.csv format to this is easy.

read_stream_data is designed to read the train set and return the wordids / wordcts
representation of the documents.

    line = f.readline() -> return Str object representing the current line, plus a trailing `\n' character.
    line = line.strip() -> remove all leading and trailing characters (space, newline, tabs etc)

## What is the format of the test set?

Same as train set.

## How is train set read?

options.seq_mode: if True, process batches of documents sequentially, from one big corpus.

train_file = file(options.data_path) -> train_file is a File object.

c = read_stream_data(train_file, batchsize) -> read the next batchsize lines of train_file, 
                                               interpreting each line as a document

## How is test set read?

c_test = read_data(test_data_path) -> read_data and read_stream_data are nearly identical.

# Subsampling mode

## What is the format of the train set?

Having just one .txt file like sequential mode is fine. However if the total number of 
documents is so large that it makes sense to split it into a few files, that works too. 
For instance, there could be a data directory with a few files wiki-split1.txt, wiki-split2.txt etc where 
each wiki-split1.txt has the form:
"
0 234 10 345 5 ...
1 684 57 452 3 ...
...
"
where the 0th document contains 10 occurences of the word with id 234 ... 
      the 1st document contains 57 occurences of the word with id 684 ... 
      ...
The data_path command line argument should be "wiki-split*"; glob will figure out the rest.