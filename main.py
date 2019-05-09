from gensim.models import KeyedVectors
import numpy as np

wv_path = "GoogleNews-vectors-negative300.bin"
test_path = "word-test.v1.txt"
relations = ["capital-world", "currency", "city-in-state", "family",
            "gram1-adjective-to-adverb", "gram2-opposite", "gram3-comparative", "gram6-nationality-adjective"]


def cos_sim(v0, v1):
    return np.dot(v0, v1) / (np.linalg.norm(v0)*np.linalg.norm(v1))


def find_analogies(w1, w2, w3):
    params = [w1, w2, w3]
    for word in params:
        if word not in word_vectors:
          print("KeyError:", word, "not in vocabulary")
          return ''

    a = word_vectors[w1]
    b = word_vectors[w2]
    c = word_vectors[w3]
    v0 = b - a + c

    max_sim = 0
    matched_word = ''

    for word in word_vectors.vocab:
        v1 = word_vectors[word]
        similarity = cos_sim(v0, v1)
        if similarity > max_sim:
            if word not in params:
                max_sim = similarity
                matched_word = word

    print("similarity:", matched_word, max_sim)
    # print(w2, "-", w1, "=", matched_word, "-", w3)

    return matched_word


def test():
    num_of_correct = 1
    total_num = 1
    flag = False
    with open(test_path) as file:
        next(file)

        for line in file:
            if line.split()[0] == ":":
                continue

            words = line.split()
            predicted_word = find_analogies(words[0], words[1], words[2])

            if words[3] == predicted_word:
                num_of_correct += 1
                total_num += 1
            else:
                # print(words[1], "-", words[0], "=", predicted_word, "-", words[2])
                total_num += 1

        #print("# of Correct Found Tags:", num_of_correct, "\n# of Total Words", total_num)
        print("Accuracy:", (100*num_of_correct) / total_num)


def get_test_set():
    test_set = set()
    with open(test_path) as file:
        next(file)

        for line in file:
            if line.split()[0] == ":":
                # print(line)
                continue

            words = line.split()
            test_set.add(words[0])
            test_set.add(words[1])
            test_set.add(words[2])
            test_set.add(words[3])

    return test_set

word_vectors = KeyedVectors.load_word2vec_format(wv_path, limit=50000, binary=True)
word_vectors.syn0norm = word_vectors.syn0
word_vectors.init_sims(replace=True)
word_vectors.save("GoogleNews")

word_vectors = KeyedVectors.load('GoogleNews', mmap='r')  # mmap the large matrix as read-only
word_vectors.syn0norm = KeyedVectors.syn0  # no need to call init_sims

# test_set = get_test_set()
# wv_dict = dict()
#
# for word in word_vectors.vocab:
#     wv_dict[word] = word_vectors[word]

# print(len(wv_dict))
# print(len(test_set))
test()
