import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.mixture import GMM


def cluster_nn(latent1, label1, latent2, k):
    clf = neighbors.KNeighborsClassifier(k, weights='distance')
    clf.fit(latent1, label1)
    inferred_labels = clf.predict(latent2)
    return inferred_labels


def cluster_accuracy_nn(real_latent1, label1, real_latent2, label2, k=4):
    inferred_labels = cluster_nn(real_latent1, label1, real_latent2, k)
    for label in range(inferred_labels.shape[0]):
        if inferred_labels[label] == 6:
            inferred_labels[label] = 5
    clustering_accuracy = np.mean(inferred_labels == label2)*100
    return clustering_accuracy, inferred_labels


def cluster_gmm(latent1, label1, latent2):
    classifier = GMM(n_components=np.unique(label1).shape[0], covariance_type='full')
    classifier.means_ = np.array([latent1[label1 == i].mean(axis=0)
                                  for i in range(np.unique(label1).shape[0])])
    classifier.fit(latent1)
    inferred_labels = classifier.predict(latent2)
    return inferred_labels


def cluster_accuracy_gmm(real_latent1, label1, real_latent2, label2):
    inferred_labels = cluster_gmm(real_latent1, label1, real_latent2)
    for label in range(inferred_labels.shape[0]):
        if inferred_labels[label] == 6:
            inferred_labels[label] = 5
    clustering_accuracy = np.mean(inferred_labels == label2)*100

    return clustering_accuracy, inferred_labels
