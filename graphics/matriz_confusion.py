import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def graficar(etiquetas, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=etiquetas)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=True,
                cbar_kws={"orientation": "vertical", "label": "color bar"},
                xticklabels=etiquetas, yticklabels=etiquetas,
                square=True, annot_kws={"size": 6})
    plt.title('Matriz de Confusión', fontsize=10)
    plt.xlabel('Predicciones', fontsize=10)
    plt.ylabel('Valores verdaderos', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

