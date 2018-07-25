# -*- coding: utf-8 -*-
"""Main module."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson, kl_divergence as kl

from scvi.metrics.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI
from scvi.models.utils import one_hot

torch.backends.cudnn.benchmark = True


# VAE model
class VAEF(nn.Module):
    r"""Variational auto-encoder model.

    Args:
        :n_input: Number of input genes for scRNA-seq data.
        :n_input_fish: Number of input genes for smFISH data
        :n_batch: Default: ``0``.
        :n_labels: Default: ``0``.
        :n_hidden: Number of hidden. Default: ``128``.
        :n_latent: Default: ``1``.
        :n_layers: Number of layers. Default: ``1``.
        :dropout_rate: Default: ``0.1``.
        :dispersion: Default: ``"gene"``.
        :log_variational: Default: ``True``.
        :reconstruction_loss: Default: ``"zinb"``.
        :reconstruction_loss_fish: Default: ``"poisson"``.

    Examples:
        >>> gene_dataset_seq = CortexDataset()
        >>> gene_dataset_fish = SmfishDataset()
        >>> vae = VAE(gene_dataset_seq.nb_genes, gene_dataset_fish.nb_genes,
        ... n_labels=gene_dataset.n_labels, use_cuda=True )

    """

    def __init__(self, n_input, n_input_fish, genes_to_discard=None, n_batch=0, n_labels=0, n_hidden=128, n_latent=10,
                 n_layers=1, dropout_rate=0.3,
                 dispersion="gene", log_variational=True, reconstruction_loss="zinb", reconstruction_loss_fish="poisson"):
        super(VAEF, self).__init__()

        self.n_input = n_input
        self.n_input_fish = n_input_fish - len(genes_to_discard)
        self.indexes_to_keep = np.arange(n_input_fish)
        self.indexes_to_keep = np.delete(self.indexes_to_keep, genes_to_discard)

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_loss_fish = reconstruction_loss_fish
        # Automatically desactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent_layers = 1

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, ))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        # First layer of the encoder isn't shared
        self.z_encoder = Encoder(n_input, n_hidden, n_hidden=n_hidden, n_layers=1,
                                 dropout_rate=dropout_rate)
        self.z_encoder_fish = Encoder(self.n_input_fish, n_hidden, n_hidden=n_hidden, n_layers=1,
                                      dropout_rate=dropout_rate)
        # The last layers of the encoder are shared
        self.z_final_encoder = Encoder(n_hidden, n_latent, n_hidden=n_hidden, n_layers=n_layers,
                                       dropout_rate=dropout_rate)

        self.decoder = DecoderSCVI(n_latent, n_input, n_layers=n_layers, n_hidden=n_hidden, n_cat_list=[2],
                                   dropout_rate=dropout_rate)

    def get_latents(self, x, y=None):
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, mode="scRNA"):
        x = torch.log(1 + x)
        # First layer isn't shared
        if mode == "scRNA":
            z, _, _ = self.z_encoder(x)
        elif mode == "smFISH":
            z, _, _ = self.z_encoder_fish(x[:, self.indexes_to_keep])
        # The last layers of the encoder are shared
        qz_m, qz_v, z = self.z_final_encoder(z)
        if not self.training:
            z = qz_m
        return z

    def get_sample_scale(self, x, mode="scRNA", batch_index=None, y=None):
        z = self.sample_from_posterior_z(x, y, mode)  # y only used in VAEC
        px = self.decoder.px_decoder(z, batch_index, y)  # y only used in VAEC
        px_scale = self.decoder.px_scale_decoder(px)
        return px_scale

    def get_sample_rate(self, x, y=None, mode="scRNA"):
        if mode == "scRNA":
            library = torch.log(torch.sum(x, dim=1)).view(-1, 1)
            batch_index = torch.zeros_like(library)
        else:
            library = torch.log(torch.sum(x[:, self.indexes_to_keep], dim=1)).view(-1, 1)
            batch_index = torch.ones_like(library)
        px_scale = self.get_sample_scale(x, batch_index=batch_index, y=y)
        return px_scale * torch.exp(library)

    def _reconstruction_loss(self, x, px_rate, px_r, px_dropout, batch_index, y, mode="scRNA"):
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        # Reconstruction Loss
        if mode == "scRNA":
            if self.reconstruction_loss == 'zinb':
                reconst_loss = -log_zinb_positive(x, px_rate, torch.exp(px_r), px_dropout)
            elif self.reconstruction_loss == 'nb':
                reconst_loss = -log_nb_positive(x, px_rate, torch.exp(px_r))
        elif mode == "smFISH":
            if self.reconstruction_loss_fish == 'poisson':
                reconst_loss = -torch.sum(Poisson(px_rate).log_prob(x), dim=1)
            elif self.reconstruction_loss_fish == 'gaussian':
                reconst_loss = -torch.sum(Normal(px_rate, 1).log_prob(x), dim=1)
        return reconst_loss

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None, mode="scRNA"):
        # Parameters for z latent distribution
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)
        # Sampling
        if mode == "scRNA":
            qz_m, qz_v, z = self.z_encoder(x_)
            library = torch.log(torch.sum(x, dim=1)).view(-1, 1)
            batch_index = torch.zeros_like(library)
        if mode == "smFISH":
            qz_m, qz_v, z = self.z_encoder_fish(x_[:, self.indexes_to_keep])
            library = torch.log(torch.sum(x[:, self.indexes_to_keep], dim=1)).view(-1, 1)
            batch_index = torch.ones_like(library)
        qz_m, qz_v, z = self.z_final_encoder(qz_m)
        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index)

        # rescaling the expected frequencies
        if mode == "smFISH":
            x = x[:, self.indexes_to_keep]
            px_scale = px_scale[:, :self.n_input_fish] / torch.sum(px_scale[:, :self.n_input_fish], dim=1).view(-1, 1)
            px_rate = px_scale * torch.exp(library)

        reconst_loss = self._reconstruction_loss(x, px_rate, px_r, px_dropout, batch_index, y, mode)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
        kl_divergence = kl_divergence_z

        return reconst_loss, kl_divergence
