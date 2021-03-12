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

    x, y = [], []
    for w in words:
        index = word2Ind[w]
        x.append(M_reduced[index][0])
        y.append(M_reduced[index][1])

    plt.plot(x, y, 'x', color='r')

    for i, xy in enumerate(zip(x, y)):
        plt.annotate("%s" % words[i], xy=xy, xytext=(0, 5), textcoords='offset points', color="k")

    plt.savefig("./2.jpg")  


def load_word2vec():
    #  Load Word2Vec Vectors
    #     Return:
    #         wv_from_bin: All 3 million embeddings, each lengh 300
    
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):
    #  Put the word2vec vectors into a matrix M.
    #     Param:
    #         wv_from_bin: KeyedVectors object; the 3 million word2vec vectors loaded from file
    #     Return:
    #         M: numpy matrix shape (num words, 300) containing the vectors
    #         word2Ind: dictionary mapping each word to its row number in M
    
    import random
    words = list(wv_from_bin.vocab.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind


def toy(): 
    wv_from_bin = load_word2vec()
    M, word2Ind = get_matrix_of_vectors(wv_from_bin)
    M_reduced = reduce_to_k_dim(M, k=2)

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    plot_embeddings(M_reduced, word2Ind, words)

    similar_word = 'leaves'
    top_similar = wv_from_bin.most_similar(similar_word)
    for w, s in top_similar:
        print("'{}: {:8f}".format(w, s))

    w1 = "happy"
    w2 = "cheerful"
    w3 = "sad"
    w1_w2_dist = wv_from_bin.distance(w1, w2)
    w1_w3_dist = wv_from_bin.distance(w1, w3)

    print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
    print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

    pprint.pprint(wv_from_bin.most_similar(positive=['China', 'Beijing'], negative=['Canada']))
    print()

    pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'boss'], negative=['man']))
    print()
    pprint.pprint(wv_from_bin.most_similar(positive=['man', 'boss'], negative=['woman']))


if __name__ == '__main__':
    toy()