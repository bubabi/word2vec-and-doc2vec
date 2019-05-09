from gensim.models import KeyedVectors
from collections import defaultdict
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
          # print("KeyError:", word, "not in vocabulary")
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

    # print("similarity:", matched_word, max_sim)
    # print(w2, "-", w1, "=", matched_word, "-", w3)

    return matched_word


def test():
    test_dict = defaultdict(list)
    current_section = ""
    total_true = 0
    total_question = 0
    flag = False
    with open(test_path) as file:
        next(file)

        for line in file:
            if line.split()[0] == ":":
                if line.split()[1] in relations:
                    current_section = line.split()[1]
                    flag = True
                    continue
                else:
                    flag = False
                    continue

            if flag:
                test_dict[current_section].append(line)

        for k, v in test_dict.items():
            print(k)
            num_of_correct = 0
            total_num = 0
            for line in v:
                words = line.split()
                predicted_word = find_analogies(words[0], words[1], words[2])

                if words[3] == predicted_word:
                    num_of_correct += 1
                    total_num += 1
                else:
                    # print(words[1], "-", words[0], "=", predicted_word, "-", words[2])
                    total_num += 1
            total_true += num_of_correct
            total_question += total_num
            print("# of correctly answered questions:", num_of_correct, "\n# of questions attempted", total_num)
            print("Accuracy:", (num_of_correct) / total_num, "\n")

        print("TOTAL: # of correctly answered questions:", total_true, "\nTOTAL: # of questions attempted", total_question)
        print("TOTAL: Accuracy:", (total_true) / total_question, "\n")


word_vectors = KeyedVectors.load_word2vec_format(wv_path, limit=50000, binary=True)


# find_analogies("son", "daughter", "king")

test()
