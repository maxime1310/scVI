import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.mixture import GMM


def cluster_nn(latent1, label1, latent2, k):
    clf = neighbors.KNeighborsClassifier(k, weights='distance')
    clf.fit(latent1, label1)
    inferred_labels = clf.predict(latent2)
    return inferred_labels


def cluster_accuracy_nn(real_latent1, latent1, label1, gene_exp_1, real_latent2, latent2, label2, gene_exp_2, k):
    inferred_labels = cluster_nn(real_latent1, label1, real_latent2, k)
    for label in range(inferred_labels.shape[0]):
        if inferred_labels[label] == 6:
            inferred_labels[label] = 5
    w = np.ones(label2.shape[0])
    from sklearn.metrics import accuracy_score

    for idx, i in enumerate(np.bincount(label2)):
        w[label2 == idx] *= (i / float(label2.shape[0]))
    print("weighted accuracy")
    print(accuracy_score(label2, inferred_labels, sample_weight=w))
    clustering_accuracy = np.mean(inferred_labels == label2)*100
    print("Clustering accuracy NN")
    print(clustering_accuracy)
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.title("Real labels")
    # plt.scatter(latent2[:, 0], latent2[:, 1], c =label2.ravel())
    plt.scatter(latent2[:, 0][label2 == 0], latent2[:, 1][label2 == 0], c='g', edgecolors='none',
                label='astro ')
    plt.scatter(latent2[:, 0][label2 == 2], latent2[:, 1][label2 == 2], c='r', edgecolors='none',
                label='inter ')
    plt.scatter(latent2[:, 0][label2 == 4], latent2[:, 1][label2 == 4], c='b', edgecolors='none',
                label='oligo ')
    plt.scatter(latent2[:, 0][label2 == 1], latent2[:, 1][label2 == 1], c='k', edgecolors='none',
                label='endothelial')
    plt.scatter(latent2[:, 0][label2 == 3], latent2[:, 1][label2 == 3], c='m', edgecolors='none',
                label='microglia ')
    plt.scatter(latent2[:, 0][label2 == 5], latent2[:, 1][label2 == 5], c='y', edgecolors='none',
                label='pyramidal')
    plt.scatter(latent2[:, 0][label2 == 6], latent2[:, 1][label2 == 6], c='y', edgecolors='none',
                label='pyramidal')
    plt.subplot(212)
    plt.title("Inferred labels")
    plt.scatter(latent2[:, 0][inferred_labels == 0], latent2[:, 1][inferred_labels == 0], c='g', edgecolors='none',
                label='astro ')
    plt.scatter(latent2[:, 0][inferred_labels == 2], latent2[:, 1][inferred_labels == 2], c='r', edgecolors='none',
                label='inter ')
    plt.scatter(latent2[:, 0][inferred_labels == 4], latent2[:, 1][inferred_labels == 4], c='b', edgecolors='none',
                label='oligo ')
    plt.scatter(latent2[:, 0][inferred_labels == 1], latent2[:, 1][inferred_labels == 1], c='k', edgecolors='none',
                label='endothelial')
    plt.scatter(latent2[:, 0][inferred_labels == 3], latent2[:, 1][inferred_labels == 3], c='m', edgecolors='none',
                label='microglia ')
    plt.scatter(latent2[:, 0][inferred_labels== 5], latent2[:, 1][inferred_labels == 5], c='y', edgecolors='none',
                label='pyramidal')
    plt.scatter(latent2[:, 0][inferred_labels == 6], latent2[:, 1][inferred_labels == 6], c='y', edgecolors='none',
                label='pyramidal')
    # plt.scatter(latent2[:, 0], latent2[:, 1], c=inferred_labels.ravel())
    plt.tight_layout()
    plt.savefig('NN_clustering.svg')
    return clustering_accuracy, inferred_labels


def cluster_gmm(latent1, label1, latent2):
    classifier = GMM(n_components=np.unique(label1).shape[0], covariance_type='tied')
    classifier.means_ = np.array([latent1[label1 == i].mean(axis=0)
                                  for i in range(np.unique(label1).shape[0])])
    classifier.fit(latent1)
    y_train_pred = classifier.predict(latent1)
    train_accuracy = np.mean(y_train_pred.ravel() == label1.ravel()) * 100
    print(train_accuracy)

    inferred_labels = classifier.predict(latent2)
    return inferred_labels


def cluster_accuracy_gmm(real_latent1, latent1, label1, gene_exp_1, real_latent2, latent2, label2, gene_exp_2, k):
    inferred_labels = cluster_gmm(real_latent1, label1, real_latent2, k)
    for label in range(inferred_labels.shape[0]):
        if inferred_labels[label] == 6:
            inferred_labels[label] = 5
    w = np.ones(label2.shape[0])
    from sklearn.metrics import accuracy_score

    for idx, i in enumerate(np.bincount(label2)):
        w[label2 == idx] *= (i / float(label2.shape[0]))
    print("weighted accuracy")
    print(accuracy_score(label2, inferred_labels, sample_weight=w))
    #build_similarity_matrix_data(gene_exp_2[:, 1:], gene_exp_2[:, 1:], inferred_labels, inferred_labels, title='Correlation of inferred labels.png')
    #build_similarity_matrix_data(gene_exp_2[:, 1:], gene_exp_2[:, 1:], label2, label2, title='Correlation of real labels.png')
    clustering_accuracy = np.mean(inferred_labels == label2)*100
    print("Clustering accuracy GMM")
    print(clustering_accuracy)
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.title("Real labels")
    # plt.scatter(latent2[:, 0], latent2[:, 1], c =label2.ravel())
    plt.scatter(latent2[:, 0][label2 == 0], latent2[:, 1][label2 == 0], c='g', edgecolors='none',
                label='astro ')
    plt.scatter(latent2[:, 0][label2 == 2], latent2[:, 1][label2 == 2], c='r', edgecolors='none',
                label='inter ')
    plt.scatter(latent2[:, 0][label2 == 4], latent2[:, 1][label2 == 4], c='b', edgecolors='none',
                label='oligo ')
    plt.scatter(latent2[:, 0][label2 == 1], latent2[:, 1][label2 == 1], c='k', edgecolors='none',
                label='endothelial')
    plt.scatter(latent2[:, 0][label2 == 3], latent2[:, 1][label2 == 3], c='m', edgecolors='none',
                label='microglia ')
    plt.scatter(latent2[:, 0][label2 == 5], latent2[:, 1][label2 == 5], c='y', edgecolors='none',
                label='pyramidal')
    plt.scatter(latent2[:, 0][label2 == 6], latent2[:, 1][label2 == 6], c='y', edgecolors='none',
                label='pyramidal')
    plt.subplot(212)
    plt.title("Inferred labels")
    plt.scatter(latent2[:, 0][inferred_labels == 0], latent2[:, 1][inferred_labels == 0], c='g', edgecolors='none',
                label='astro ')
    plt.scatter(latent2[:, 0][inferred_labels == 2], latent2[:, 1][inferred_labels == 2], c='r', edgecolors='none',
                label='inter ')
    plt.scatter(latent2[:, 0][inferred_labels == 4], latent2[:, 1][inferred_labels == 4], c='b', edgecolors='none',
                label='oligo ')
    plt.scatter(latent2[:, 0][inferred_labels == 1], latent2[:, 1][inferred_labels == 1], c='k', edgecolors='none',
                label='endothelial')
    plt.scatter(latent2[:, 0][inferred_labels == 3], latent2[:, 1][inferred_labels == 3], c='m', edgecolors='none',
                label='microglia ')
    plt.scatter(latent2[:, 0][inferred_labels== 5], latent2[:, 1][inferred_labels == 5], c='y', edgecolors='none',
                label='pyramidal')
    plt.scatter(latent2[:, 0][inferred_labels == 6], latent2[:, 1][inferred_labels == 6], c='y', edgecolors='none',
                label='pyramidal')
    # plt.scatter(latent2[:, 0], latent2[:, 1], c=inferred_labels.ravel())
    plt.tight_layout()
    plt.savefig('GMM_clustering.svg')
    return clustering_accuracy, inferred_labels
