import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score
from torch.nn import functional as F

from scvi.dataset import CortexDataset
from scvi.dataset.data_loaders import DataLoaders
from smFISHxscRNA.dataset.data_loaders import TrainTestDataLoadersFish
from scvi.metrics.classification import compute_accuracy, compute_accuracy_svc, compute_accuracy_rf
from scvi.metrics.clustering import entropy_batch_mixing
from smFISHxscRNA.metrics.clustering import get_latent
from smFISHxscRNA.metrics.log_likelihood import compute_log_likelihood
from . import InferenceFish

plt.switch_backend('agg')


class VariationalInferenceFish(InferenceFish):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SVAEC``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset_seq = CortexDataset()
        >>> gene_dataset_fish = SmfishDataset()
        >>> vae = VAE(gene_dataset_seq.nb_genes, gene_dataset_fish.nb_genes,
        ... n_labels=gene_dataset.n_labels, use_cuda=True)

        >>> infer = VariationalInference(gene_dataset_seq, gene_dataset_fish, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, gene_dataset_seq, gene_dataset_fish, train_size=0.8, use_cuda=True, cl_ratio=0,
                 fish_ponderation=1, **kwargs):
        super(VariationalInferenceFish, self).__init__(model, gene_dataset_seq, gene_dataset_fish, use_cuda=use_cuda, **kwargs)
        self.kl = None
        self.cl_ratio = cl_ratio
        self.fish_ponderation = fish_ponderation
        self.data_loaders = TrainTestDataLoadersFish(self.gene_dataset_seq, self.gene_dataset_fish,
                                                     train_size=train_size, use_cuda=use_cuda)

    def loss(self, tensors_seq, tensors_fish):
        sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors_seq
        reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var, batch_index, mode="scRNA")
        reconst_loss += self.cl_ratio * F.cross_entropy(self.model.classify(sample_batch, mode="scRNA"), labels.view(-1))
        loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)
        sample_batch_fish, local_l_mean, local_l_var, batch_index_fish, _, _, _ = tensors_fish
        reconst_loss_fish, kl_divergence_fish = self.model(sample_batch_fish, local_l_mean, local_l_var, batch_index_fish, mode="smFISH")
        loss_fish = torch.mean(reconst_loss_fish + self.kl_weight * kl_divergence_fish)
        loss = loss * sample_batch.size(0) + loss_fish * sample_batch_fish.size(0) * self.fish_ponderation
        loss /= (sample_batch.size(0) + sample_batch_fish.size(0))
        return loss + loss_fish

    def on_epoch_begin(self):
        self.kl_weight = self.kl if self.kl is not None else min(1, 1/4000 * self.epoch / self.n_epochs)

    def ll(self, name, verbose=True):
        if name == 'train_seq' or name == 'test_seq':
            ll = compute_log_likelihood(self.model, self.data_loaders[name], mode="scRNA")
        if name == 'train_fish' or name == 'test_fish':
            ll = compute_log_likelihood(self.model, self.data_loaders[name], mode="smFISH")
        if verbose:
            print("LL for %s is : %.4f" % (name, ll))
        return ll

    ll.mode = 'min'

    def clustering_scores(self, name, verbose=True):
        if self.gene_dataset.n_labels > 1:
            latent, _, labels = get_latent(self.model, self.data_loaders[name])
            labels_pred = KMeans(self.gene_dataset.n_labels, n_init=200).fit_predict(latent)  # n_jobs>1 ?
            asw_score = silhouette_score(latent, labels)
            nmi_score = NMI(labels, labels_pred)
            ari_score = ARI(labels, labels_pred)
            if verbose:
                print("Clustering Scores for %s:\nSilhouette: %.4f\nNMI: %.4f\nARI: %.4f" %
                      (name, asw_score, nmi_score, ari_score))
            return asw_score, nmi_score, ari_score

    def entropy_batch_mixing(self, name, verbose=False, **kwargs):
        if self.gene_dataset.n_batches == 2:
            latent, batch_indices, labels = get_latent(self.model, self.data_loaders[name])
            be_score = entropy_batch_mixing(latent, batch_indices, **kwargs)
            if verbose:
                print("Entropy batch mixing :", be_score)
            return be_score

    entropy_batch_mixing.mode = 'max'

    def show_t_sne(self, name, n_samples=1000, color_by='', save_name=''):
        latent, batch_indices, labels = get_latent(self.model, self.data_loaders[name])
        idx_t_sne = np.random.permutation(len(latent))[:n_samples] if n_samples else np.arange(len(latent))
        if latent.shape[1] != 2:
            latent = TSNE().fit_transform(latent[idx_t_sne])
        if not color_by:
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1])
        else:
            batch_indices = batch_indices[idx_t_sne].ravel()
            labels = labels[idx_t_sne].ravel()
            if color_by == 'batches' or color_by == 'labels':
                indices = batch_indices if color_by == 'batches' else labels
                n = self.gene_dataset.n_batches if color_by == 'batches' else self.gene_dataset.n_labels
                if hasattr(self.gene_dataset, 'cell_types') and color_by == 'labels':
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(len(np.unique(indices)))]
                plt.figure(figsize=(10, 10))
                for i, label in zip(range(n), plt_labels):
                    plt.scatter(latent[indices == i, 0], latent[indices == i, 1], label=label)
                plt.legend()
            elif color_by == 'batches and labels':
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                for i in range(self.gene_dataset.n_batches):
                    axes[0].scatter(latent[batch_indices == i, 0], latent[batch_indices == i, 1], label=str(i))
                axes[0].set_title("batch coloring")
                axes[0].axis("off")
                axes[0].legend()

                indices = labels
                if hasattr(self.gene_dataset, 'cell_types'):
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(len(np.unique(indices)))]
                for i, cell_type in zip(range(self.gene_dataset.n_labels), plt_labels):
                    axes[1].scatter(latent[indices == i, 0], latent[indices == i, 1], label=cell_type)
                axes[1].set_title("label coloring")
                axes[1].axis("off")
                axes[1].legend()
        plt.axis("off")
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)
