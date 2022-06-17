"""Practical 1

Greatly inspired by Stanford CS224 2019 class.
"""

import sys

import pprint

import matplotlib.pyplot as plt
import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import random
import nltk

nltk.download('reuters')
nltk.download('pl196x')
import random

import numpy as np
import scipy as sp
from nltk.corpus import reuters
from nltk.corpus.reader import pl196x
from sklearn.decomposition import PCA, TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


# TODO: a)
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the 
            corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the 
            corpus
    """
    corpus_words = []
    num_corpus_words = 0

    for lst1 in corpus:
        for word in lst1:
            if word not in corpus_words:
                corpus_words.append(word)
                num_corpus_words += 1

    return sorted(corpus_words), num_corpus_words

# ---------------------


# TODO: b)
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window.
            Words near edges will have a smaller number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the 
                same as the ordering of the words given by the distinct_words 
                function.
            word2Ind (dict): dictionary that maps word to index 
                (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = {word: i for i, word in enumerate(words)}

    for lst1 in corpus:
        for i, word in enumerate(lst1):
            context = lst1[max(i-window_size,0): i] + lst1[i+1: i+1+window_size]
            for contex_word in context:
                M[word2Ind[word]][word2Ind[contex_word]] += 1

    return M, word2Ind

# ---------------------


# TODO: c)
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality
        (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following
         SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number 
                of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)):
            matrix of k-dimensioal word embeddings.
            In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    M_reduced =svd.fit_transform(M)

    print("Done.")
    return M_reduced

# ---------------------


# TODO: d)
def plot_embeddings(M_reduced, word2Ind, words):
    from adjustText import adjust_text
    import matplotlib
    """ Plot in a scatterplot the embeddings of the words specified 
        in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the
            corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to
            visualize
    """

    proper_idxs = [word2Ind[word] for word in words]
    x, y = zip(*M_reduced[proper_idxs])
    with plt.style.context("ggplot"):
        plt.rcParams.update({'font.size': 8})
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        x_width = max(x) - min(x)
        y_width = max(y) - min(y)
        plt.xlim([min(x)-x_width/50, max(x)+x_width/50])
        plt.ylim([min(y)-y_width/50, max(y)+y_width/50])
        for i, word in enumerate(words):
            ax.annotate(word, (x[i], y[i]))
        annotations = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Annotation)]
        adjust_text(annotations)
        plt.show()

# ---------------------


# TODO: e)
# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------

def read_corpus_pl():
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    pl196x_dir = nltk.data.find('corpora/pl196x')
    pl = pl196x.Pl196xCorpusReader(
        pl196x_dir, r'.*\.xml', textids='textids.txt',cat_file="cats.txt")
    tsents = pl.tagged_sents(fileids=pl.fileids(),categories='cats.txt')[:5000]

    return [[START_TOKEN] + [
        w[0].lower() for w in list(sent)] + [END_TOKEN] for sent in tsents]


def plot_unnormalized(corpus, words):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    plot_embeddings(M_reduced_co_occurrence, word2Ind_co_occurrence, words)


def plot_normalized(corpus, words):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths.reshape(-1, 1) # broadcasting
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words)


#################################
# Section 2:
#################################
# Then run the following to load the word2vec vectors into memory. 
# Note: This might take several minutes.
# wv_from_bin_pl = KeyedVectors.load("word2vec/word2vec_100_3_polish.bin")

# -----------------------------------
# Run Cell to Load Word Vectors
# Note: This may take several minutes
# -----------------------------------


#################################
# TODO: a)
def get_matrix_of_vectors(wv_from_bin, required_words):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors
                         loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    words = list(wv_from_bin.key_to_index.keys())
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

# -----------------------------------------------------------------
# Run Cell to Reduce 300-Dimensinal Word Embeddings to k Dimensions
# Note: This may take several minutes
# -----------------------------------------------------------------

#################################
# TODO: a)

# M, word2Ind = get_matrix_of_vectors(wv_from_bin_pl, words)
# M_reduced = reduce_to_k_dim(M, k=2)
#
# plot_embeddings(M_reduced, word2Ind, words)

#################################
# TODO: b)
# Polysemous Words
# ------------------
# Write your polysemous word exploration code here.
# IN THE NOTEBOOK

# wv_from_bin_pl.most_similar("stówa")
# ------------------

#################################
# TODO: c)
# Synonyms & Antonyms
# ------------------
# Write your synonym & antonym exploration code here.
# IN THE NOTEBOOK

# w1 = "radosny"
# w2 = "pogodny"
# w3 = "smutny"
# w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
# w1_w3_dist = wv_from_bin_pl.distance(w1, w3)
#
# print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
# print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))


#################################
# TODO: d)
# Solving Analogies with Word Vectors
# ------------------
# IN THE NOTEBOOK
# ------------------
# Write your analogy exploration code here.

# pprint.pprint(wv_from_bin_pl.most_similar(
#     positive=["syn", "kobieta"], negative=["mezczyzna"]))


#################################
# TODO: e)
# Incorrect Analogy
# ------------------
# Write your incorrect analogy exploration code here.
# IN THE NOTEBOOK
# ------------------


#################################
# TODO: f)
# Guided Analysis of Bias in Word Vectors
# Here `positive` indicates the list of words to be similar to and 
# `negative` indicates the list of words to be most dissimilar from.
# ------------------
# pprint.pprint(wv_from_bin_pl.most_similar(
#     positive=['kobieta', 'szef'], negative=['mężczyzna']))
# print()
# pprint.pprint(wv_from_bin_pl.most_similar(
#     positive=['mężczyzna', 'prezes'], negative=['kobieta']))

# IN THE NOTEBOOK
#################################
# TODO: g)
# Independent Analysis of Bias in Word Vectors 
# ------------------
# IN THE NOTEBOOK

#################################
# Section 3:
# English part
#################################
def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin

# wv_from_bin = load_word2vec()

#################################
# TODO:
# IN THE NOTEBOOK
#################################
