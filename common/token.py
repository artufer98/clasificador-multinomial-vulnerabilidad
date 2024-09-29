import re
import string
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import SpanishStemmer, stopwords

lemmatizador = SpanishStemmer()
def generar(texto):
    limpiar_token = lambda token: re.sub(r'\W|\d', "", token)

    # Tokenización
    tokens = wordpunct_tokenize(texto)
    tokens = [
        limpiar_token(token)
        for token in tokens
        if (token not in string.punctuation and
            token.isdigit() == False and
            len(limpiar_token(token)) > 1)
    ]

    # Lemmatización
    # El metodo stem ya convierte el token a minuscula
    unigramas_lemmas = [lemmatizador.stem(token) for token in tokens]
    return [
        unigrama_lemma for unigrama_lemma in unigramas_lemmas
        if unigrama_lemma not in stopwords.words("spanish")
    ]


def frecuencia(tokens):
    return FreqDist(tokens)


def generar_lista_unica(documentos):
    return sorted(set().union(*(
        documento.tokens for documento in documentos)))

