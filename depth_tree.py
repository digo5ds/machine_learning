import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, tree
from sklearn.model_selection import KFold


def acuracy_analisys():
    """
    Função com o objetivo de gerar um gráfico para análise da acuracia de um modelo de treinamento de
    arvore de decisão, levando em consideração o incremento do nível de complexidade do algoritimo


    """
    dataset = open('files/dataset_small.pkl', 'rb')
    (X, y) = pickle.load(dataset, encoding='latin1')

    # Validação cruzada com a técnica K-fold, usando 10 partições
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    # Variação dos parâmetros: profundidade da árvore de decisão
    complexities = np.arange(2, 20, )

    acc = np.zeros((10, 18))
    i = 0

    # Validação cruzada
    for train_index, val_index in kf.split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        j = 0
        for c in complexities:
            dt = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=c)
            dt.fit(x_train, y_train)
            yhat = dt.predict(x_val)
            acc[i][j] = metrics.accuracy_score(yhat, y_val)
            j = j + 1
        i = i + 1

    plt.boxplot(acc)
    for i in range(18):
        xderiv = (i + 1) * np.ones(acc[:, i].shape) + \
                 (np.random.rand(10, ) - 0.5) * 0.1
        plt.plot(xderiv, acc[:, i], 'ro', alpha=0.3)

    plt.ylim((0.7, 1.))
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    plt.xlabel('Complexidade')
    plt.ylabel('Acuracia')
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    plt.savefig(os.path.join('plots', 'acuracy_analisys.png'), dpi=300, bbox_inches='tight')


def error_analisys(tree_depth):
    """"
    Função com o objetivo de gerar um gráfico para analisar a sensibilidade de um modelo de arvore de decisão,
    levando em consideração o tamanho do dataset e o nível de complexidade

    Parameters:

        tree_depth (int): complexidade da arvore de decisão

    """

    maxn = 1000
    vector_test = np.zeros((10, 299, 2))
    vector_train = np.zeros((10, 299, 2))

    for i in range(10):
        vector_position = 0

        # gera o conjunto de treiamento
        x = np.concatenate([1.25 * np.random.randn(maxn, 2),
                            5 + 1.5 * np.random.randn(maxn, 2)])
        x = np.concatenate([x, [8, 5] + 1.5 * np.random.randn(maxn, 2)])
        y = np.concatenate([np.ones((maxn, 1)), -np.ones((maxn, 1))])
        y = np.concatenate([y, np.ones((maxn, 1))])
        perm = np.random.permutation(y.size)
        dataset_train = [x[perm, :], y[perm]]

        # gera o conjunto de treinamento
        x_test = np.concatenate(
            [1.25 * np.random.randn(maxn, 2), 5 + 1.5 * np.random.randn(maxn, 2)])
        x_test = np.concatenate(
            [x_test, [8, 5] + 1.5 * np.random.randn(maxn, 2)])
        y_test = np.concatenate([np.ones((maxn, 1)), -np.ones((maxn, 1))])
        y_test = np.concatenate([y_test, np.ones((maxn, 1))])

        for N in range(10, 3000, 10):
            xr = dataset_train[0][:N, :]
            yr = dataset_train[1][:N]
            # criando classificador
            clf = tree.DecisionTreeClassifier(
                min_samples_leaf=1, max_depth=tree_depth)
            clf.fit(xr, yr.ravel())
            # armazena métricas do erro de teste
            vector_test[i, vector_position, 0] = 1. - metrics.accuracy_score(clf.predict(x_test), y_test.ravel())
            # armazena métricas do erro de treinamento
            vector_train[i, vector_position, 0] = 1. - metrics.accuracy_score(clf.predict(xr), yr.ravel())
            vector_position += 1

    test_error, = plt.plot(np.mean(vector_train[:, :, 0].T, axis=1), 'pink')
    train_error, = plt.plot(np.mean(vector_test[:, :, 0].T, axis=1), 'c')
    fig = plt.gcf()
    fig.set_size_inches(12, 5)

    plt.xlabel('Numero de exemplos x10')
    plt.ylabel('Taxa de erro')
    plt.legend([test_error, train_error], ['Teste com profundidade = {}'.format(
        tree_depth), 'Treinamento com profundidade = {}'.format(tree_depth)])

    if not os.path.isdir('plots'):
        os.mkdir('plots')
    plt.savefig(os.path.join('plots', 'error_analisys_{}.png'.format(tree_depth)), dpi=300, bbox_inches='tight')
    plt.close()
