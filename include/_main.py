# /usr/bin/env python3

from random import shuffle
import re
import string
import contractions
from pprint import pprint
from nltk import LancasterStemmer, TextCollection, WordNetLemmatizer, defaultdict
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from read_file import read
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download("stopwords")

stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

def remove_digits(sentence):
    return re.sub(r'\d', "", sentence)

def char_to_whitespace(sentence):
    backlash = r'\/'
    dot = r'\.' 
    patterns = [backlash, dot]
    regex = re.compile("|".join((pattern for pattern in patterns)))

    return re.sub(regex, " ", sentence)

def clean(sentence):
    sentence = (char_to_whitespace(remove_digits(sentence.lower()))
        .encode("ascii", errors="ignore")
        .decode())

    return contractions.fix(sentence)

def is_revelant(word):
    return (
        word not in stopwords.words("english") and
        word not in string.punctuation
    )

def process_document(document):
    return list(
        stemmer.stem(lemmatizer.lemmatize(token))
        for token in word_tokenize(clean(document))
        if is_revelant(token) == True
    )

def build_corpora(documents):
    corpora = set()
    for document in documents:
        corpora.update(process_document(document))

    return list(corpora)


def vectorize(corpora, corpus):
    corpus = [process_document(document) for document in corpus]
    texts = TextCollection(corpus)

    vectors = []
    for document in corpus:
        vector = dict.fromkeys(corpora, 0)

        for term in corpora:
            vector[term] = texts.tf_idf(term, document)    

        vectors.append(vector)

    return vectors


expert = read("./tufts-dental-database/Expert/expert.json")
expert_documents = [e['Description'] for e in expert]
shuffle(expert_documents)

student = read("./tufts-dental-database/Student/student.json")
student_documents = [e['Description'] for e in expert]
shuffle(student_documents)

corpora = build_corpora(expert_documents + student_documents)
expert_vectors = list(zip(vectorize(corpora, expert_documents), ["expert"] * expert_documents.__len__()))
student_vectors = list(zip(vectorize(corpora, student_documents), ["student"] * student_documents.__len__()))

# vectors = expert_vectors + student_vectors
# shuffle(vectors)

length = 100
# train_set, test_set = vectors[:length], vectors[length:length*2]
train_set = expert_vectors[:length] + student_vectors[:length]
test_set = expert_vectors[length:length*2] + student_vectors[length:length*2]

model = NaiveBayesClassifier.train(train_set)
print(accuracy(model, test_set))
print(model.show_most_informative_features(5))

