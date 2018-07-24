import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


def proximity_imputation(real_latent1, gene_exp_1, real_latent2, gene_exp_2, index_1, index_2, k):
    knn = neighbors.KNeighborsRegressor(k, weights='distance')
    gene_exp_1 = gene_exp_1 / (np.sum(gene_exp_1[:, :gene_exp_2.shape[1]], axis=1).reshape(-1, 1) + 1e-8)
    y = knn.fit(real_latent1, gene_exp_1[:, index_1]).predict(real_latent2)
    plt.subplot(211)
    plt.title('Differences between imputed and real values for unobserved gene')
    plt.hist(np.abs(y[:150]-gene_exp_2[:, index_2][:150]), bins=30)
    plt.tight_layout()
    plt.legend()
    plt.subplot(212)
    plt.title('Differences between imputed and real values for unobserved gene')
    plt.plot(range(gene_exp_2.shape[0])[:100], gene_exp_2[:, index_2][:100], c='b', label='real values')
    plt.plot(range(gene_exp_2.shape[0])[:100], y[:100], c='r', label='imputed values')
    plt.legend()
    plt.tight_layout()
    plt.savefig('proximity_imputation.svg')
    print("Imputation errors")
    mean_imputation_error = np.mean(np.abs(gene_exp_2[:, index_2]-y))
    print(mean_imputation_error)
    median_imputation_error = np.median(np.abs(gene_exp_2[:, index_2] - y))
    print(median_imputation_error)
    print("Relative imputation errors")
    mean_imputation_error = np.mean(np.abs(gene_exp_2[:, index_2] - y)/0.5*(gene_exp_2[:, index_2] + y))
    print(mean_imputation_error)
    median_imputation_error = np.median(np.abs(gene_exp_2[:, index_2] - y)/0.5*(gene_exp_2[:, index_2] + y))
    print(median_imputation_error)
    plot_correlation(gene_exp_2[:, index_2], y)
    return mean_imputation_error


def plot_correlation(real_values, imputed_values):
    plt.figure(figsize=(10, 10))
    plt.scatter(real_values, imputed_values, c='b')
    plt.scatter(real_values, real_values, c='r')
    plt.savefig('correlation_imputation.svg')

