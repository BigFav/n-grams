# n-grams

A pure pythonic n-gram program that allows you to generate a random sentence
from a training set, compute perplexity of a test set on a training set, and
perform a perplexity-based multi-class classification. This program requires
no outside packages, no installation, and runs on all Python 2.7 and above.

## Usage

The usage of this n-gram interface is:

    ngrams.py [-h] [-n N] -sent training_set
              [-h] [-n N] [-sent] [-t T] [-ls [α] | -gts] [-p] [-c [CATEGORY_NUM] output_file] training_set test_set

Running the command:

    ./ngrams.py -h

or

    python ngrams.py -h

will provide a help message, with some explanation for each option.

## Generate Random Sentence

You can generate a random sentence by inputting the ``-sent`` option,
and a text file. This will generate a random sentence based on an unsmoothed
n-gram model. Note: you can insert an 'n' by inserting the ``-n`` flag followed
by the desired n; if no n is inserted, n is set to 2 (bigrams).

Example:

    python ngrams.py -sent -n 4 review.train
    It is one of chicago 's best recently renovated to bring it up .

## Perplexity

You can find the perplexity of two pieces of text using the ``-p`` option, and
inserting the two text files. You can also choose which type of smoothing you
would like to use: Laplace (Additive) or Good Turing smoothing. If using
Laplacian smoothing, you can set the smoothing parameter by adding a float
value after the ``-ls`` option. Note: Good Turing smoothing has only been
implemented for unigrams and bigrams. Also, Laplacian smoothing is used by
default. In addition, this program parses texts slightly differently if it
knows that it will be classified.

Example:

    ./ngrams.py -ls .2 -p kjbible.train kjbible.test
    Perplexity: 216.38605101456963

The perplexity will slightly depend on the Python version, as the math module
was updated in Python 3.x.

## Multi-Class Classification

You can classify text a pieces of text by providing a training set and the test
set you wish to classify. The examples provided in the test set will have their
perplexities compared to every class in the training set in order to classify
each example.

Example:

    ./ngrams.py -c classifications.txt reviews.train reviews.test

## Putting it All Together

You can do all of these operations at once. Note: the train and test sets must
be at the end of every command, in that order.

Example:

    ./ngrams.py -n 2 -sent -ls .2 -p -c 1 classes.txt reviews.train reviews.test
    When I complained I asked for us to be transferred to the Sheraton which they arranged for us .
    Perplexity: 218.65148036450822
