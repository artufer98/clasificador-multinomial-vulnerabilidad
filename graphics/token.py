import pandas as pd
import matplotlib.pyplot as plt
from seaborn import swarmplot


def graficar(tabla):
    datos = pd.melt(
        tabla[["madr", "nna", "sujet", "etiqueta"]],
        id_vars="etiqueta",
        var_name="tokens",
        value_name="valor")
    swarmplot(data=datos, x="tokens", y="valor", hue="etiqueta", size=4)
    plt.show()
