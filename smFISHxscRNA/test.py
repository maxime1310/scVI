from smFISHxscRNA.dataset import CortexDataset, SmfishDataset
from smFISHxscRNA.inference import VariationalInferenceFish
from smFISHxscRNA.models import VAEF
from smFISHxscRNA.metrics.clustering import get_data, get_common_t_sne, entropy_batch_mixing
from smFISHxscRNA.metrics.visualisation import show_cell_types, show_mixing, compare_cell_types, show_gene_exp, show_spatial_expression
from smFISHxscRNA.metrics.classification import cluster_accuracy_nn
from smFISHxscRNA.metrics.imputation import plot_correlation, proximity_imputation, compute_metrics, get_index
import numpy as np
from smFISHxscRNA.train import train_FISHVAE_jointly

genes_to_discard = ['gad2']
gene_dataset_fish = SmfishDataset()
gene_names = gene_dataset_fish.gene_names
l = []
for n_gene in range(len(gene_names)):
    for gene in genes_to_discard:
        if gene_names[n_gene].lower() == gene.lower():
            l.append(n_gene)
genes_to_discard = l
# The "genes_to_discard" argument is given here so that the order of the genes in CortexDataset matches
# the order in SmfishDataset
gene_dataset_seq = CortexDataset(genes_fish=gene_dataset_fish.gene_names, genes_to_discard=genes_to_discard,
                                 genes_to_keep=["mog", "sst", "gja1", "ctss"], additional_genes=500)

# Getting indexes of genes to ignore during training (to validate the imputation task)
print(gene_dataset_seq.gene_names)
print(gene_dataset_fish.gene_names)
indexes_to_keep = np.arange(len(gene_dataset_fish.gene_names))
indexes_to_keep = np.delete(indexes_to_keep, genes_to_discard)
print(gene_dataset_fish.gene_names[indexes_to_keep])

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
use_cuda = False
gene_dataset = gene_dataset_seq
tt_split =0.9
example_indices = np.random.permutation(len(gene_dataset))
tt_split = int(tt_split * len(gene_dataset))  # 90%/10% train/test split
data_loader_seq = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,
                               sampler=SubsetRandomSampler(example_indices[:tt_split]),
                               collate_fn=gene_dataset.collate_fn)


gene_dataset = gene_dataset_fish
tt_split =0.9
example_indices = np.random.permutation(len(gene_dataset))
tt_split = int(tt_split * len(gene_dataset))  # 90%/10% train/test split
data_loader_fish = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,
                               sampler=SubsetRandomSampler(example_indices[:tt_split]),
                               collate_fn=gene_dataset.collate_fn)

vae = VAEF(gene_dataset_seq.nb_genes, gene_dataset_fish.nb_genes, indexes_fish_train=genes_to_discard, n_latent=8,
           n_layers=2, n_hidden=300, reconstruction_loss='nb', dropout_rate=0.4)
train_FISHVAE_jointly(vae, data_loader_seq, data_loader_fish, lr=0.008, n_epochs=1)
#infer = VariationalInferenceFish(vae, gene_dataset_seq, gene_dataset_fish, train_size=0.9, verbose=True, frequency=5)
#infer.train(n_epochs=200, lr=0.0008)
#
#data_loader_fish = infer.data_loaders['train_fish']
#data_loader_seq = infer.data_loaders['train_seq']

latent_seq, _, labels_seq, expected_frequencies_seq, values_seq = get_data(vae, data_loader_seq, mode="scRNA")
latent_fish, _, labels_fish, expected_frequencies_fish, values_fish, x_coords, y_coords = get_data(vae, data_loader_fish, mode="smFISH")

t_sne_seq, t_sne_fish, idx_t_sne_seq, idx_t_sne_fish = get_common_t_sne(latent_seq, latent_fish, n_samples=1000)
show_cell_types(t_sne_seq, labels_seq[idx_t_sne_seq], t_sne_fish, labels_fish[idx_t_sne_fish])
show_mixing(t_sne_seq, t_sne_fish)
print(entropy_batch_mixing(np.concatenate((t_sne_seq, t_sne_fish)),
                           batches=np.concatenate((np.zeros_like(idx_t_sne_seq),
                                                  np.ones_like(idx_t_sne_fish)))))
accuracy, inferred_labels = cluster_accuracy_nn(latent_seq, labels_seq, latent_fish, labels_fish)
print(accuracy)
compare_cell_types(t_sne_fish, labels_fish[idx_t_sne_fish], inferred_labels[idx_t_sne_fish])

normed_expected_frequencies_seq = expected_frequencies_seq / np.sum(expected_frequencies_seq[:, :vae.n_input_fish],
                                                                    axis=1).reshape(-1, 1)
normed_expected_frequencies_fish = expected_frequencies_fish / np.sum(expected_frequencies_fish[:, :vae.n_input_fish],
                                                                    axis=1).reshape(-1, 1)
print(normed_expected_frequencies_fish.shape)
print(expected_frequencies_fish.shape)
print(np.sum(expected_frequencies_fish[:, :vae.n_input_fish], axis=1).reshape(-1, 1).shape)
idx_to_impute = get_index(gene_dataset_seq.gene_names, "gad2")
idx_astro = get_index(gene_dataset_seq.gene_names, "gja1")
idx_oligo = get_index(gene_dataset_seq.gene_names, "mog")
idx_interneurons = get_index(gene_dataset_seq.gene_names, "sst")

proximity_imputed_values_fish = proximity_imputation(latent_seq, normed_expected_frequencies_seq[:, idx_to_impute],
                                                     latent_fish)

real_values_fish = values_fish / np.sum(values_fish, axis=1).reshape(-1, 1)
real_values_fish = real_values_fish[:, 0]

plot_correlation(real_values_fish, proximity_imputed_values_fish)
plot_correlation(real_values_fish, normed_expected_frequencies_fish[:, idx_to_impute])

print(compute_metrics(real_values_fish, proximity_imputed_values_fish))
print(compute_metrics(real_values_fish, normed_expected_frequencies_fish[:, idx_to_impute]))

show_gene_exp(t_sne_fish, normed_expected_frequencies_fish[idx_t_sne_fish, idx_astro], labels=labels_fish[idx_t_sne_fish],
              title="latent_exp_astro.svg", gene_name="gja1")
show_gene_exp(t_sne_fish, normed_expected_frequencies_fish[idx_t_sne_fish, idx_oligo], labels=labels_fish[idx_t_sne_fish],
              title="latent_exp_oligo.svg", gene_name="mog")
show_gene_exp(t_sne_fish, normed_expected_frequencies_fish[idx_t_sne_fish, idx_interneurons], labels=labels_fish[idx_t_sne_fish],
              title="latent_exp_interneurons.svg", gene_name="sst")

show_spatial_expression(x_coords[idx_t_sne_fish], y_coords[idx_t_sne_fish],
                        normed_expected_frequencies_fish[idx_t_sne_fish, idx_astro], labels=labels_fish[idx_t_sne_fish],
                        title="spatial_exp_astro.svg", gene_name="gja1")
show_spatial_expression(x_coords[idx_t_sne_fish], y_coords[idx_t_sne_fish],
                        normed_expected_frequencies_fish[idx_t_sne_fish, idx_oligo], labels=labels_fish[idx_t_sne_fish],
                        title="spatial_exp_oligo.svg", gene_name="mog")
show_spatial_expression(x_coords[idx_t_sne_fish], y_coords[idx_t_sne_fish],
                        normed_expected_frequencies_fish[idx_t_sne_fish, idx_interneurons],
                        labels=labels_fish[idx_t_sne_fish], title="spatial_exp_interneurons.svg", gene_name="sst")
