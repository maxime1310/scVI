import numpy as np
from sklearn import neighbors
from scipy.stats import kde
import matplotlib.pyplot as plt


def plot_imputation(imputed, original):
    x, y = imputed, original
    ymax = np.amax(y)
    mask = x < ymax
    x = x[mask]
    y = y[mask]
    mask = y < ymax
    x = x[mask]
    y = y[mask]
    data = np.vstack([x, y])
    plt.figure(figsize=(5, 5))
    axes = plt.gca()
    axes.set_xlim([0, ymax])
    axes.set_ylim([0, ymax])
    nbins = 150
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    xi, yi = np.mgrid[0:ymax:nbins * 1j, 0:ymax:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.title('Imputation', fontsize=12)
    plt.ylabel("Imputed counts")
    plt.xlabel('Original counts')

    plt.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds")

    a, _, _, _ = np.linalg.lstsq(y[:, np.newaxis], x)
    l = np.linspace(0, ymax)
    plt.plot(l, a * l, color='black')

    plt.plot(l, l, color='black', linestyle=":")


def proximity_imputation(real_latent1, normed_gene_exp_1, real_latent2, k=4):
    knn = neighbors.KNeighborsRegressor(k, weights='distance')
    y = knn.fit(real_latent1, normed_gene_exp_1).predict(real_latent2)
    return y


def plot_correlation(real_values, imputed_values):
    plt.figure(figsize=(10, 10))
    plt.scatter(real_values, imputed_values, c='b')
    plt.scatter(real_values, real_values, c='r')
    plt.savefig('correlation_imputation.svg')


def compute_metrics(real_frequencies, imputed_frequencies):
    mean_imputation_error = np.mean(np.abs((imputed_frequencies - real_frequencies)))
    median_imputation_error = np.median(np.abs((imputed_frequencies - real_frequencies)))
    relative_mean_imputation_error = np.mean(
        np.abs(imputed_frequencies - real_frequencies) / 0.5 * (imputed_frequencies + real_frequencies))
    relative_median_imputation_error = np.median(
        np.abs(imputed_frequencies - real_frequencies) / 0.5 * (imputed_frequencies + real_frequencies))
    return mean_imputation_error, median_imputation_error, relative_mean_imputation_error, \
              relative_median_imputation_error


def get_index(gene_names, gene):
    idx = 0
    for gene_cortex in range(len(gene_names)):
        if gene_names[gene_cortex].lower() == gene.lower():
            idx = gene_cortex
            print("Found idx " + str(idx) + " for gene " + gene + "!")
    return idx
