import logging
import random
import numpy as np
import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, LabeledSentence
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
import csv
import multiprocessing
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt')

cores = multiprocessing.cpu_count()
tags_index = {'sci-fi': 1, 'action': 2, 'comedy': 3, 'fantasy': 4, 'animation': 5, 'romance': 6}
train_documents = []
test_documents = []
x_train = []
x_test = []
y_train = []
y_test = []
i = 0


# Function for tokenizing
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


# Reading the file
with open('tagged_plots_movielens.csv', 'r') as csvfile:
    next(csvfile)
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for line in csv_reader:
        tokenized_text = tokenize_text(line[2])
        tag = tags_index.get(line[3])
        if i <= 2000:
            train_documents.append(TaggedDocument(words=tokenized_text,tags=[tag]))
            x_train.append(tokenized_text)
            y_train.append(tag)
        else:
            test_documents.append(TaggedDocument(words=tokenized_text, tags=[tag]))
            x_test.append(tokenized_text)
            y_test.append(tag)
        i += 1


def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(LabeledSentence(v, [label]))
    return labeled

# Function for tokenizing
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

# def train():
#     d2v = Doc2Vec(min_count=1,
#                   window=5,
#                   vector_size=300,
#                   workers=cores,
#                   alpha=0.025,
#                   dm=1)
#
#     d2v.build_vocab(train_documents)
#     print("Training")
#     # random.shuffle(train_documents)
#
#     d2v.train(train_documents, total_examples=len(train_documents), epochs=10)
#     d2v.save("d2v.model")
#

def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):

    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors


def train_classifier(d2v, training_vectors, training_labels):
    logging.info("Classifier training")
    train_vectors = get_vectors(d2v, len(training_vectors), 300, 'Train')
    model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model

x_train = label_sentences(x_train, 'Train')
x_test = label_sentences(x_test, 'Test')
all_data = x_train + x_test

print("Building Doc2Vec vocabulary")
d2v_model = Doc2Vec(min_count=1, window=5, vector_size=300, workers=multiprocessing.cpu_count(),
                      alpha=0.025, min_alpha=0.00025, dm=0, dm_mean=1, dm_tag_count=1)
d2v_model.build_vocab(all_data)

print("Training Doc2Vec model")
for epoch in range(5):
    print('Training iteration #{0}'.format(epoch))
    d2v_model.train(all_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
    d2v_model.alpha -= 0.0002
    d2v_model.min_alpha = d2v_model.alpha

print("Saving trained Doc2Vec model")
d2v_model.save("d2v.model")

#d2v_model = Doc2Vec.load("d2v.model")
print("Model Loaded")
classifier = train_classifier(d2v_model, x_train, y_train)

