from gensim.models import KeyedVectors


wv_from_bin_pl = KeyedVectors.load("word2vec/word2vec_100_3_polish.bin")


wv_from_bin_pl.most_similar("twierdzenie")


wv_from_bin_pl.most_similar("zamek")


wv_from_bin_pl.most_similar("bank")


wv_from_bin_pl.most_similar("osobnik")


wv_from_bin_pl.most_similar("siła")


wv_from_bin_pl.most_similar("zero")


w1 = 'daleki'
w2 = 'odległy'
w3 = 'bliski'

w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
w1_w3_dist = wv_from_bin_pl.distance(w1, w3)


print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))


import pprint


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["syn", "kobieta"], negative=["mezczyzna"]))


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["wagon", "ręka"], negative=["pociąg"]))


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['jeden', "fałsz"], negative=["prawda"]))


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["pierwszy", "przedostatni"], negative=["drugi"]))


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'szef'], negative=['mężczyzna']))


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['mężczyzna', 'prezes'], negative=['kobieta']))


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'żołnierz'], negative=['mężczyzna']))


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['polak', 'praca'], negative=['włoch']))


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['włoch', 'praca'], negative=['polak']))


def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print("Loaded vocab size get_ipython().run_line_magic("i"", " % len(vocab))")
    return wv_from_bin

wv_from_bin = load_word2vec()


wv_from_bin.most_similar("proposition")


wv_from_bin.most_similar("castle")


wv_from_bin.most_similar("bank")


wv_from_bin_pl.most_similar("osobnik")


wv_from_bin_pl.most_similar("power")


wv_from_bin_pl.most_similar("zero")


w1 = 'far'
w2 = 'distant'
w3 = 'close'

w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
w1_w3_dist = wv_from_bin_pl.distance(w1, w3)


print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))


first : final :: second :: penultimate


pprint.pprint(wv_from_bin.most_similar(
    positive=["first", "penultimate"], negative=["second"]))


pprint.pprint(wv_from_bin.most_similar(
    positive=["false", 'one'], negative=["true"]))


pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'boss'], negative=['man']))


pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'soldier'], negative=['man']))



