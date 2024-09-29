import matplotlib.pyplot as plt
from common import documento, corpus
from graphics import matriz_confusion, etiqueta, token
from models import naive_bayes
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, recall_score


documentos = documento.procesar_varios()
_corpus = corpus.generar(documentos)
tabla = corpus.construir_tabla(_corpus)
etiqueta.graficar(tabla)
token.graficar(tabla)
etiquetas = tabla["etiqueta"].unique()

X_train, X_test, y_train, y_test = corpus.construir_conjuntos(tabla)

y_pred = naive_bayes.predecir(X_train, X_test, y_train)

print(f"PRECISION: {naive_bayes.precision(y_test, y_pred)}")
print(f"RECALL: {naive_bayes.recall(y_test, y_pred)}")
f1_score_all = round(f1_score(y_test, y_pred, average="weighted"), 3)
print(f"F1-SCORE SIN SELECCION DE CARACTERISTICAS: {f1_score_all}")

matriz_confusion.graficar(etiquetas, y_test, y_pred)

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

f1_score_list = []
modelo = naive_bayes.obtener()

x_f1_score_list=list(range(1,14))

for k in x_f1_score_list:
    print(k)
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_train_v2, y_train_v2)

    sel_X_train_v2 = selector.transform(X_train_v2)
    sel_X_test_v2 = selector.transform(X_test_v2)

    modelo.fit(sel_X_train_v2, y_train_v2)
    kbest_preds = modelo.predict(sel_X_test_v2)

    f1_score_kbest = round(f1_score(y_test_v2, kbest_preds, average="weighted"), 3)
    f1_score_list.append(f1_score_kbest)

print(f1_score_list)
print(x_f1_score_list)

plt.bar(x_f1_score_list, f1_score_list, color='skyblue')
plt.xlabel('Nro de Caracteristicas')
plt.ylabel('F1-Score')
plt.title('F1-Scores')
plt.show()

def obtener_indice_max_f1(f1_scores):
    # Obtener el índice del F1-score más alto
    indice_max = f1_scores.index(max(f1_scores))
    # Sumar 1 al índice para convertirlo en un índice basado en 1
    return indice_max + 1

k = obtener_indice_max_f1(f1_score_list)
print(k)
selector = SelectKBest(mutual_info_classif, k=k)
selector.fit(X_train_v2, y_train_v2)

sel_X_train_v2 = selector.transform(X_train_v2)
sel_X_test_v2 = selector.transform(X_test_v2)

modelo.fit(sel_X_train_v2, y_train_v2)
y_pred_v2 = modelo.predict(sel_X_test_v2)

f1_score_all_v2 = round(f1_score(y_test_v2, y_pred_v2, average='weighted'), 3)
print(f"F1-SCORE CON SELECCION DE CARACTERISTICAS: {f1_score_all}")

plt.bar(["SIN SELECCION", "CON SELECCION"], [f1_score_all, f1_score_all_v2], color='skyblue')
plt.xlabel('Nro de Caracteristicas')
plt.ylabel('F1-Score')
plt.title('F1-Scores')
plt.show()

# selected_feature_mask = selector.get_support()
#
# selected_features = X_train_v2.columns[selected_feature_mask]
#
# print(selected_features)
