import pandas as pd


def relatos_etiquetados():
    registros = pd.read_excel('datos.xlsx', usecols=["vulneracion", "relato"])
    registros = registros.to_dict(orient="list")
    etiquetas = registros["vulneracion"]
    relatos = registros["relato"]

    return zip(etiquetas, relatos)

