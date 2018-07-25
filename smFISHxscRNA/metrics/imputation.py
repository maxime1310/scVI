import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


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
