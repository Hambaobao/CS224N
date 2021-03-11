# All Import Statements Defined Here
# Note: Do not add to this list.
# All the dependencies you need, can be installed by running .

import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
 
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


def read_corpus(category="crude"):
    # Read files from the specified Reuter's category.
    #     Params:
    #         category (string): category name
    #     Return:
    #         list of lists, with words from each of the processed files
    
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]


def distinct_words(corpus):
    # Determine a list of distinct words for the corpus.
    #     Params:
    #         corpus (list of list of strings): corpus of documents
    #     Return:
    #         corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
    #         num_corpus_words (integer): number of distinct words across the corpus
    
    corpus_words_list = [w for l in corpus for w in l]
    corpus_words = sorted(list(set(corpus_words_list)))
    
    num_corpus_words = len(corpus_words)

    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    #  Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
    #     Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
    #           number of co-occurring words.
              
    #           For example, if we take the document "START All that glitters is not gold END" with window size of 4,
    #           "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
    #     Params:
    #         corpus (list of list of strings): corpus of documents
    #         window_size (int): size of context window
    #     Return:
    #         M (numpy matrix of shape (number of corpus words, number of corpus words)): 
    #             Co-occurence matrix of word counts. 
    #             The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
    #         word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    
    words, num_words = distinct_words(corpus)
    
    M = np.zeros((num_words, num_words))
    
    word2Ind =  {w: i for w, i in zip(words, range(num_words))}

    for sentence in corpus:
        for index, word in enumerate(sentence):
            for i in range(-window_size, window_size + 1):
                if i and index + i > -1 and index + i < len(sentence):
                    M[word2Ind[word]][word2Ind[sentence[index + i]]] += 1

    return M, word2Ind


def reduce_to_k_dim(M, k=2):
    #  Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
    #     to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
    #         - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
    #     Params:
    #         M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
    #         k (int): embedding size of each word after dimension reduction
    #     Return:
    #         M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
    #                 In terms of the SVD from math class, this actually returns U * S
        
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`

    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    
    svd.fit(M)

    M_reduced = svd.transform(M)

    print("Done.")

    return M_reduced


def plot_embeddings(M_reduced, word2Ind, words):
    #  Plot in a scatterplot the embeddings of the words specified in the list "words".
    #     NOTE: do not plot all the words listed in M_reduced / word2Ind.
    #     Include a label next to each point.
        
    #     Params:
    #         M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): matrix of k-dimensioal word embeddings
    #         word2Ind (dict): dictionary that maps word to indices for matrix M
    #         words (list of strings): words whose embeddings we want to visualize
    
    x = M_reduced[:, 0]
    y = M_reduced[:, 1]

    plt.plot(x, y, 'x', color='r')

    for i, xy in enumerate(zip(x, y)):
        plt.annotate("%s" % words[i], xy=xy, xytext=(0, 5), textcoords='offset points', color="k")

    plt.savefig("./1.jpg")  


def toy():
    # ---------------------
    # Run this sanity check
    # Note that this not an exhaustive check for correctness.
    # The plot produced should look like the "test solution plot" depicted below. 
    # ---------------------

    print ("-" * 80)
    print ("Outputted Plot:")

    M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
    word2Ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
    words = ['test1', 'test2', 'test3', 'test4', 'test5']
    plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words)

    print ("-" * 80)


if __name__ == '__main__':
    # reuters_corpus = read_corpus()

    # compute_co_occurrence_matrix(reuters_corpus, window_size=4)

    toy()