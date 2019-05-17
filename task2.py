import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, LabeledSentence
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import csv
import multiprocessing
import matplotlib.pyplot as plt

cores = multiprocessing.cpu_count()
tags_index = {'sci-fi': 1, 'action': 2, 'comedy': 3,
              'fantasy': 4, 'animation': 5, 'romance': 6}

stop = stopwords.words('english') + list(string.punctuation)

train_documents = []
test_documents = []

x_train = []
x_test = []

y_train = []
y_test = []


def accuracy_plot(x, y_train, y_test):

    # Plot the data
    plt.plot(x, y_train, label='train accuracy')
    plt.plot(x, y_test, label='test accuracy')

    # Add a legend
    plt.legend()
    plt.title("vector_size & accuracy")
    plt.xlabel("vector_size")
    plt.ylabel("accuracy %")
    # Show the plot
    plt.show()


def preprocessing_plots(text):
    return [i for i in word_tokenize(text.lower()) if i not in stop]

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


def train_doc2vec_model(corpus, epoch_size, vector_size):
    d2v_model = Doc2Vec(min_count=1, window=10, vector_size=vector_size,
                        workers=cores, alpha=0.025, dm=0)
    d2v_model.build_vocab(corpus)

    for epoch in range(epoch_size):
        d2v_model.train(corpus,
                        total_examples=d2v_model.corpus_count,
                        epochs=d2v_model.epochs)

    return d2v_model


def get_plot_vectors(d2v_model, corpus_size, vectors_size, type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        label = type + ':' + str(i)
        vectors[i] = d2v_model.docvecs[label]
    return vectors


def train_classifier(d2v_model, training_vectors, training_labels, vector_size):
    train_vectors = get_plot_vectors(d2v_model, len(training_vectors), vector_size, 'Train')
    model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    accuracy = accuracy_score(training_labels, training_predictions)
    return model, accuracy*100


def test(d2v_model, classifier, testing_vectors, testing_labels, vector_size):
    test_vectors = get_plot_vectors(d2v_model, len(testing_vectors), vector_size, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    accuracy = accuracy_score(testing_labels, testing_predictions)
    return accuracy*100

  
read_csv()

x_train = set_tag_to_plot(x_train, 'Train')
x_test = set_tag_to_plot(x_test, 'Test')
corpus = x_train + x_test


# vector_size_list = [100, 200, 300, 400, 500]
# train_acc_list = list()
# test_acc_list = list()
#
# for vs in vector_size_list:
#     d2v_model = train_doc2vec_model(corpus, 8, vs)
#     classifier, train_accuracy = train_classifier(d2v_model, x_train, y_train, vs)
#     test_accuracy = test(d2v_model, classifier, x_test, y_test, vs)
#     train_acc_list.append(train_accuracy)
#     test_acc_list.append(test_accuracy)
#
# accuracy_plot(vector_size_list, train_acc_list, test_acc_list)

d2v_model = train_doc2vec_model(corpus, 5, 300)
classifier, train_accuracy = train_classifier(d2v_model, x_train, y_train, 300)
test_accuracy = test(d2v_model, classifier, x_test, y_test, 300)
print(train_accuracy, test_accuracy)
