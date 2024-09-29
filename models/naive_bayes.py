from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, recall_score


modelo = MultinomialNB()

def obtener():
    return modelo


def predecir(x_train, x_test, y_train):
    modelo.fit(x_train, y_train)
    return modelo.predict(x_test)


def precision(y_test, y_pred):
    return accuracy_score(y_test, y_pred)


def recall(y_test, y_pred):
    return recall_score(y_test, y_pred, average='weighted')


def reporte(y_test, y_pred):
    return classification_report(y_test, y_pred)
