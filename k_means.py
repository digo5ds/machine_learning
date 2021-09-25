import os
import numpy as np
from sklearn import cluster, metrics
import matplotlib.pylab as plt


def k_means_analisys():
    """
    Função com o objetivo de gerar uma base de treinamento genérica com o algoritmo k_means
    e também métricas para a avaliação do treinamento

    """

    k = 3
    # gerando alguns dados para teste
    maxn = 40
    x = np.concatenate([1.25 * np.random.randn(maxn, 2),
                        5 + 1.5 * np.random.randn(maxn, 2)])
    x = np.concatenate([x, [8, 3] + 1.2 * np.random.randn(maxn, 2)])

    # Apenas para propósitos de visualização, criar rótulos de 3 distribuições
    y = np.concatenate([np.ones((maxn, 1)), 2 * np.ones((maxn, 1))])
    y = np.concatenate([y, 3 * np.ones((maxn, 1))])

    clf = cluster.KMeans(init='random', n_clusters=k, random_state=0)

    clf.fit(x)
    zx = clf.predict(x)

    plt.subplot(1, 3, 1)
    plt.title('Rotulos originais', size=14)
    plt.scatter(x[(y == 1).ravel(), 0], x[(y == 1).ravel(), 1], color='r')
    plt.scatter(x[(y == 2).ravel(), 0], x[(y == 2).ravel(), 1], color='b')  # b
    plt.scatter(x[(y == 3).ravel(), 0], x[(y == 3).ravel(), 1], color='g')  # g
    fig = plt.gcf()
    fig.set_size_inches((12, 3))

    plt.subplot(1, 3, 2)
    plt.title('Dados sem os rotulos', size=14)
    plt.scatter(x[(y == 1).ravel(), 0], x[(y == 1).ravel(), 1], color='r')
    plt.scatter(x[(y == 2).ravel(), 0], x[(y == 2).ravel(), 1], color='r')  # b
    plt.scatter(x[(y == 3).ravel(), 0], x[(y == 3).ravel(), 1], color='r')  # g
    fig = plt.gcf()
    fig.set_size_inches((12, 3))

    plt.subplot(1, 3, 3)
    plt.title('Dados clusterizados', size=14)
    plt.scatter(x[(zx == 1).ravel(), 0], x[(zx == 1).ravel(), 1], color='r')
    plt.scatter(x[(zx == 2).ravel(), 0], x[(zx == 2).ravel(), 1], color='b')
    plt.scatter(x[(zx == 0).ravel(), 0], x[(zx == 0).ravel(), 1], color='g')
    fig = plt.gcf()
    fig.set_size_inches((12, 3))

    clf = cluster.KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=0,
        max_iter=300,
        n_init=10)

    clf.fit(x)  # efetuando treinamento

    print('Inertia: %.2f' % clf.inertia_)

    print(
        'Adjusted_rand_score %.2f' %
        metrics.adjusted_rand_score(
            y.ravel(),
            clf.labels_))

    # Homogeneity: valor no intervalo [0, 1] e valores próximos a 1 são
    # melhores
    print(
        'Homogeneity %.2f' %
        metrics.homogeneity_score(
            y.ravel(),
            clf.labels_))

    # Completeness: valor no intervalo [0, 1] e valores próximos a 1 são
    # melhores
    print(
        'Completeness %.2f' %
        metrics.completeness_score(
            y.ravel(),
            clf.labels_))

    # V_measure: média harmônica entre Homogeneity e Completeness; e valores
    # próximos a 1 são melhores
    print('V_measure %.2f' % metrics.v_measure_score(y.ravel(), clf.labels_))

    # Silhouette: valor no intervalo [-1, 1]; valores positivos próximos a 1
    # significam clusters compactos.
    print(
        'Silhouette %.2f' %
        metrics.silhouette_score(
            x,
            clf.labels_,
            metric='euclidean'))

    plt.show()
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    plt.savefig(os.path.join('plots', 'k_means{}.png'.format(3)))
