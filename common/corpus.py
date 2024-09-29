from . import token
import pandas as pd
from sklearn.model_selection import train_test_split

def generar(documentos):
    corpus = []
    tokens = token.generar_lista_unica(documentos)

    for documento in documentos:
        caracteristica = {}
        caracteristica["etiqueta"] = documento.etiqueta

        for _token in tokens:
            if _token in documento.frecuencias.keys():
                caracteristica[_token] = documento.frecuencias.get(_token)
            else:
                caracteristica[_token] = 0

        corpus.append(caracteristica)

    return corpus


def construir_tabla(corpus):
    return pd.DataFrame(corpus)


def construir_conjuntos(tabla):
    X = tabla.drop(["etiqueta"], axis=1)
    y = tabla["etiqueta"]

    return train_test_split(X, y, test_size=0.3, shuffle= True,
                            stratify=y, random_state=42)

