from gensim.models import KeyedVectors
import numpy as np
from numpy import linalg as LA

path = "GoogleNews-vectors-negative300.bin"

def cos_sim(v0, v1):
    return np.dot(v0, v1) / (LA.norm(v0)*LA.norm(v1))

def find_analogies(w1, w2, w3):
    params = [w1, w2, w3]
    for word in params:
        if word not in word_vectors:
          print("KeyError:", word, "not in vocabulary")
          return

    a = word_vectors[w1]
    b = word_vectors[w2]
    c = word_vectors[w3]
    v0 = b - a + c

    max_sim = 0
    matched_word = ''

    for idx, word in enumerate(word_vectors.vocab):
        if word not in params:
            v1 = word_vectors[word]
            similarity = cos_sim(v0, v1)

            if similarity > max_sim:
                max_sim = similarity
                matched_word = word

    print("similarity:", matched_word, max_sim)
    print(w2, "-", w1, "=", matched_word, "-", w3)

    return matched_word



word_vectors = KeyedVectors.load_word2vec_format(path, limit=500000, binary=True)
# result = word_vectors.most_similar(positive=['king', 'he'], negative=['queen'])
# print("{}: {:.4f}".format(*result[0]))

find_analogies('Athens', 'Greece', 'Rome')
