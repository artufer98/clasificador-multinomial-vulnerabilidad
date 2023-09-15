# /usr/bin/env python3

import json
import re
import string
import nltk
import numpy
import contractions
from functools import reduce
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import snowball
from collections import namedtuple

nltk.download("stopwords")
nltk.download("punkt")

stemmer = snowball.EnglishStemmer()


def build_corpora(sentences):
    corpora = set()
    for sentence in sentences:
        corpora.update(process_sentence(sentence))

    return list(corpora)


def remove_digits(sentence):
    return re.sub(r'\d', "", sentence)


def char_to_whitespace(sentence):
    backlash = r'\/'
    dot = r'\.' 
    patterns = [backlash, dot]
    regex = re.compile("|".join((pattern for pattern in patterns)))

    return re.sub(regex, " ", sentence)


def fix_apostrophe(sentence):
    sentence = re.sub(r'’', "'", sentence)
    sentence = re.sub(r'”|“', '', sentence)
    return sentence


def remove_connector(sentence):
    def build_connector(pattern, connector, text):
        return {
            "match": re.search(pattern[connector], sentence),
            "replace": lambda: re.sub(pattern[connector], text, sentence)
        }

    pattern = {
        "year-old": r'year-old',
        "pack-year": r'pack-year',
    }

    connectors = {
        "year-old": build_connector(pattern, "year-old", " year old "),
        "pack-year": build_connector(pattern, "pack-year", " pack per year "),
    }

    for connector in connectors.values():
        if connector["match"]:
            sentence = connector["replace"]()

    return sentence


def expand_abbreviation(sentence):
    def build_pattern(text):
        start_with = rf"^{re.escape(text)}\s"
        between_whitespace = rf"\s{re.escape(text)}\s"
        start_parenthesis = rf"\({re.escape(text)}\s"
        between_parenthesis = rf"\({re.escape(text)}\)"
        end_parenthesis = rf"\s{re.escape(text)}\)"
        end_with = rf"\s{re.escape(text)}$"
        patterns = [
            start_with,
            between_whitespace,
            start_parenthesis,
            between_parenthesis,
            end_parenthesis,
            end_with
        ]

        return re.compile("|".join((pattern for pattern in patterns)))


    def build_abbrevation(pattern, abbrevation, text):
        return {
            "match": re.search(pattern[abbrevation], sentence),
            "replace": lambda: re.sub(pattern[abbrevation], text, sentence)
        }


    pattern = {
        "ml": build_pattern("ml"),
        "mg": build_pattern("mg"),
        "mm": build_pattern("mm"),
        "copd": build_pattern("copd"),
        "ct": build_pattern("ct"),
        "spect": build_pattern("spect"),
        "egfr": build_pattern("egfr"),
        "fda": build_pattern("fda"),
        "fev": build_pattern("fev"),
        "fvc": build_pattern("fvc"),
        "iu": build_pattern("iu"),
        "iv": build_pattern("iv"),
        "kg": build_pattern("kg"),
        "pcr": build_pattern("pcr"),
        "tt": build_pattern("tt"),
        "hrt": build_pattern("hrt"),
        "cm": build_pattern("cm"),
        "bmi": build_pattern("bmi"),
    }

    abbrevations = {
        "ml": build_abbrevation(pattern, "ml", " milliliter "),
        "mg": build_abbrevation(pattern, "mg", " milligram "),
        "mm": build_abbrevation(pattern, "mm", " millimeter "),
        "copd": build_abbrevation(pattern, "copd",
                                  " chronic obstructive pulmonary disease "),
        "ct": build_abbrevation(pattern, "ct", " computed tomography "),
        "spect": build_abbrevation(
            pattern, "spect", " single photon emission computed tomography "),
        "egfr": build_abbrevation(pattern, "egfr",
                                  " epidermal growth factor receptor "),
        "fda": build_abbrevation(pattern, "fda",
                                 " food and drug Administration "),
        "fev": build_abbrevation(pattern, "fev",
                                 " forced expiratory volume "),
        "fvc": build_abbrevation(pattern, "fvc", " forced vital capacity "),
        "iu": build_abbrevation(pattern, "iu", " international unit "),
        "iv": build_abbrevation(pattern, "iv", " intravenous "),
        "kg": build_abbrevation(pattern, "kg", " kilogram "),
        "pcr": build_abbrevation(pattern, "pcr",
                                 " polymerase chain reaction "),
        "tt": build_abbrevation(pattern, "tt", " treatment "),
        "hrt": build_abbrevation(pattern, "hrt",
                                 " hormone replacement therapy "),
        "cm": build_abbrevation(pattern, "cm", " centimeter "),
        "bmi": build_abbrevation(pattern, "bmi", " body mass index "),
    }

    for abbrevation in abbrevations.values():
        if abbrevation["match"]:
            sentence = abbrevation["replace"]()

    return sentence


def clean(sentence):
    return expand_abbreviation(
        remove_connector(
            fix_apostrophe(
                char_to_whitespace(
                    remove_digits(sentence)
                )
            )
        )
    ).encode("ascii", errors="ignore").decode()


def normalize(sentence):
    return contractions.fix(clean(sentence.lower()))


def is_revelant(word):
    irrelevant_words = ["'s", "–"]

    return (
        word not in stopwords.words("english") and
        word not in string.punctuation and
        word not in irrelevant_words
    )


def remove_characters(stem):
    def build_pattern(text):
        start_with = rf"^{re.escape(text)}"
        end_with = rf"{re.escape(text)}$"
        patterns = [start_with, end_with]

        return re.compile("|".join((pattern for pattern in patterns)))

    hyphen = build_pattern("-")
    equal = build_pattern("=")
    patterns = [hyphen, equal]

    for pattern in patterns:
        stem = re.sub(pattern, "", stem)

    return stem


def process_sentence(sentence):
    return list(
        remove_characters(stemmer.stem(word))
        for word in word_tokenize(normalize(sentence))
        if is_revelant(word) == True
    )


def build_frequency(corpora, sentences, labels, frequency={}):
    Frequency = namedtuple("Frequency", ["word", "label"])

    for sentence, label in zip(sentences, labels):
        processed_sentence = process_sentence(sentence)

        for word in corpora:
            pair = Frequency(word, label)

            if pair in frequency:
                frequency[pair] += processed_sentence.count(word)
            else:
                frequency[pair] = processed_sentence.count(word)

    return frequency


def to_fixed(number):
    return float(str(round(number, 3)))


def sentence_probability(labeled_sentences, total_sentences):
    return len(labeled_sentences) / len(total_sentences)


def ratio_probability(positive_sentences, negative_sentences):
    return (
        numpy.log(len(positive_sentences)) -
        numpy.log(len(negative_sentences)))


def count_label(count, pair, frequency, label):
    if pair.label == label:
        count += frequency[pair]

    return count


def laplacian_smoothing(frequency, v):
    total_positives = reduce(
        lambda count, pair: count_label(count, pair, frequency, 1),
        frequency, 0)
    total_negatives = reduce(
        lambda count, pair: count_label(count, pair, frequency, 0),
        frequency, 0)

    for pair in frequency:
        if pair.label == 1:
            frequency[pair] = (frequency[pair]+1) / (total_positives+v)
        else:
            frequency[pair] = (frequency[pair]+1) / (total_negatives+v)

    return frequency


def get_log_likelihood(frequency, corpora):
    Frequency = namedtuple("Frequency", ["word", "label"])
 
    likelihood_frequency = {}
    for word in corpora:
        positive_freq = frequency[Frequency(word, 1)]
        negative_freq = frequency[Frequency(word, 0)]
        probability = numpy.log(positive_freq/negative_freq)
        likelihood_frequency[word] = probability

    return likelihood_frequency


def train_naive_bayes_classifier(
        positive_sentences, negative_sentences, sentences, labels):
    corpora = build_corpora(sentences)
    dict_frequency = build_frequency(corpora, sentences, labels)
    smoothed_dict_frequency = laplacian_smoothing(dict_frequency,
                                                  len(corpora))

    return (
        to_fixed(ratio_probability(positive_sentences, negative_sentences)),
        get_log_likelihood(smoothed_dict_frequency, corpora))


def naive_bayes_classifier(sentence, log_prior, log_likehood):
    sentence = process_sentence(sentence)

    probability = log_prior
    for word in sentence:
        if word in log_likehood:
            probability += log_likehood[word]

    return to_fixed(probability)


def test_naive_bayes_classifier(sentences, labels, log_prior, log_likehood):
    predicted_labels = [
        1 if naive_bayes_classifier(sentence, log_prior, log_likehood) > 0 else 0
        for sentence in sentences
    ]

    scores = [
        1 if label_x == label_y else 0
        for label_x, label_y in zip(predicted_labels, labels)
    ]

    accuracy = (numpy.sum(scores) / len(labels))

    return to_fixed(accuracy * 100)


def build_labels(sentences, label):
    return [label for _ in sentences]


def labels(positive_sentences, negative_sentences):
    return (build_labels(positive_sentences, 1) +
            build_labels(negative_sentences, 0))


def read_sentences(filename):
    with open(filename, "r") as file:
        return json.load(file)


all_positive_sentences = read_sentences("./positive-notes.json")
all_negative_sentences = read_sentences("./negative-notes.json")

test_positive_sentences = all_positive_sentences[11:]
train_positive_sentences = all_positive_sentences[:11]

test_negative_sentences = all_negative_sentences[11:]
train_negative_sentences = all_negative_sentences[:11]

train_sentences = train_positive_sentences + train_negative_sentences
test_sentences = test_positive_sentences + test_negative_sentences

train_labels = labels(train_positive_sentences, train_negative_sentences)
test_labels = labels(test_positive_sentences, test_negative_sentences)

log_prior, log_likehood = train_naive_bayes_classifier(
    train_positive_sentences, train_negative_sentences, train_sentences,
    train_labels)

