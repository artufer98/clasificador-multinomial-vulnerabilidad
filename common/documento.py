from . import token, lector, documento, etiquetador
from collections import namedtuple

def generar(etiqueta, texto=""):
    Documento = namedtuple(
        "Documento",
        ["etiqueta", "tokens", "frecuencias", "texto"])

    tokens = token.generar(texto)
    return Documento(
        etiqueta,
        set(tokens),
        token.frecuencia(tokens),
        texto)


def procesar_varios():
    documentos = []
    for (etiqueta, relato) in lector.relatos_etiquetados(): 
        texto = ''
        if type(relato) == str:
            texto = relato
        else:
            texto = documentos[-1].texto

        documentos.append(documento.generar(
            etiquetador.procesar(etiqueta), texto))

    return documentos

