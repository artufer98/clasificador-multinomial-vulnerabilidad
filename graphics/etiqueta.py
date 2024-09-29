import matplotlib.pyplot as plt


def graficar(tabla):
    items = tabla["etiqueta"].value_counts()
    _, ax = plt.subplots()

    x = list(items.keys())
    y = list(items.values)

    ax.bar(x, y, width=0.5)
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Cantidad")
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=5.5)

    for index, value in enumerate(y):
        plt.text(x=int(index), y=value+1, s=str(value), ha="center")

    plt.tight_layout()
    plt.show()

