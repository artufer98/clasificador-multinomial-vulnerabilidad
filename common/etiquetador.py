def procesar(etiqueta):
    etiqueta = etiqueta.lower().strip()
    if etiqueta == "consultas legales":
        etiqueta = "asuntos legales"
    elif etiqueta == "trabajo infantil":
        etiqueta = "explotación comercial"
    elif (etiqueta == "nna en situación de calle" or
        etiqueta == "situaciones de riesgo" or
        etiqueta == "relacionados a la vivienda"):
        etiqueta = "situación de riesgo"
    elif etiqueta == "situaciones de salud":
        etiqueta = "salud"

    return etiqueta
