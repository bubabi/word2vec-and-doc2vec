import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, LabeledSentence
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import csv
import multiprocessing
import nltk
nltk.download('punkt')

cores = multiprocessing.cpu_count()
tags_index = {'sci-fi': 1, 'action': 2, 'comedy': 3,
              'fantasy': 4, 'animation': 5, 'romance': 6}

train_documents = []
test_documents = []

x_train = []
x_test = []

y_train = []
y_test = []


def preprocessing_plots(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def read_csv():
    with open('tagged_plots_movielens.csv', 'r') as csvfile:
        next(csvfile)
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        i = 0
        for line in csv_reader:
            tokenized_text = preprocessing_plots(line[2])
            tag = tags_index.get(line[3])
            if i <= 2000:
                train_documents.append(TaggedDocument(words=tokenized_text, tags=[tag]))
                x_train.append(tokenized_text)
                y_train.append(tag)
            else:
                test_documents.append(TaggedDocument(words=tokenized_text, tags=[tag]))
                x_test.append(tokenized_text)
                y_test.append(tag)
            i += 1


def set_tag_to_plot(corpus, type):
    tagged = []
    for i, v in enumerate(corpus):
        tag = type + ':' + str(i)
        tagged.append(LabeledSentence(v, [tag]))
    return tagged


def train_doc2vec_model(corpus):
    d2v_model = Doc2Vec(min_count=1, window=10, vector_size=500,
                        workers=cores, alpha=0.025, dm=0)
    d2v_model.build_vocab(corpus)

    for epoch in range(10):
        print('Training iteration #{0}'.format(epoch))
        d2v_model.train(corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

    return d2v_model


def get_plot_vectors(d2v_model, corpus_size, vectors_size, type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        label = type + ':' + str(i)
        vectors[i] = d2v_model.docvecs[label]
    return vectors


def train_classifier(d2v_model, training_vectors, training_labels):
    train_vectors = get_plot_vectors(d2v_model, len(training_vectors), 500, 'Train')
    model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    print('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    return model


def test(d2v_model, classifier, testing_vectors, testing_labels):
    test_vectors = get_plot_vectors(d2v_model, len(testing_vectors), 500, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    print('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))


read_csv()

x_train = set_tag_to_plot(x_train, 'Train')
x_test = set_tag_to_plot(x_test, 'Test')
corpus = x_train + x_test

d2v_model = train_doc2vec_model(corpus)
classifier = train_classifier(d2v_model, x_train, y_train)
test(d2v_model, classifier, x_test, y_test)
