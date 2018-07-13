import numpy as np
import torch

from scvi.utils import to_cuda, no_grad, eval_modules


@no_grad()
def de_stats(vae, data_loader, M_sampling=100):
    """
    Output average over statistics in a symmetric way (a against b)
    forget the sets if permutation is True
    :param vae: The generative model and encoder network
    :param data_loader: a data loader for a particular dataset
    :param M_sampling: number of samples
    :return: A 1-d vector of statistics of size n_genes
    """
    px_scales = []
    all_labels = []
    for tensors in data_loader:
        if vae.use_cuda:
            tensors = to_cuda(tensors)
        sample_batch, _, _, batch_index, labels = tensors
        sample_batch = sample_batch.type(torch.float32)
        sample_batch = sample_batch.repeat(1, M_sampling).view(-1, sample_batch.size(1))
        batch_index = batch_index.repeat(1, M_sampling).view(-1, 1)
        labels = labels.repeat(1, M_sampling).view(-1, 1)
        px_scales += [vae.get_sample_scale(sample_batch, y=labels, batch_index=batch_index).cpu()]
        all_labels += [labels.cpu()]

    px_scale = torch.cat(px_scales)
    all_labels = torch.cat(all_labels)

    return px_scale, all_labels


@no_grad()
@eval_modules()
def de_cortex(px_scale, all_labels, gene_names, M_permutation=100000, permutation=False):
    """
    Output average over statistics in a symmetric way (a against b)
    forget the sets if permutation is True
    :param M_permutation: 10000 - default value in Romain's code
    :param permutation:
    :return: A 1-d vector of statistics of size n_genes
    """
    # Compute sample rate for the whole dataset ?
    cell_types = np.array(['astrocytes_ependymal', 'endothelial-mural', 'interneurons', 'microglia',
                           'oligodendrocytes', 'pyramidal CA1', 'pyramidal SS'], dtype=np.str)
    # oligodendrocytes (#4) VS pyramidal CA1 (#5)
    couple_celltypes = (4, 5)  # the couple types on which to study DE

    print("\nDifferential Expression A/B for cell types\nA: %s\nB: %s\n" %
          tuple((cell_types[couple_celltypes[i]] for i in [0, 1])))

    # Here instead of A, B = 200, 400: we do on whole dataset then select cells
    sample_rate_a = (px_scale[all_labels.view(-1) == couple_celltypes[0]].view(-1, px_scale.size(1))
                     .cpu().detach().numpy())
    sample_rate_b = (px_scale[all_labels.view(-1) == couple_celltypes[1]].view(-1, px_scale.size(1))
                     .cpu().detach().numpy())

    # agregate dataset
    samples = np.vstack((sample_rate_a, sample_rate_b))

    # prepare the pairs for sampling
    list_1 = list(np.arange(sample_rate_a.shape[0]))
    list_2 = list(sample_rate_a.shape[0] + np.arange(sample_rate_b.shape[0]))
    if not permutation:
        # case1: no permutation, sample from A and then from B
        u, v = np.random.choice(list_1, size=M_permutation), np.random.choice(list_2, size=M_permutation)
    else:
        # case2: permutation, sample from A+B twice
        u, v = (np.random.choice(list_1 + list_2, size=M_permutation),
                np.random.choice(list_1 + list_2, size=M_permutation))

    # then constitutes the pairs
    first_set = samples[u]
    second_set = samples[v]

    res = np.mean(first_set >= second_set, 0)
    res = np.log(res + 1e-8) - np.log(1 - res + 1e-8)

    genes_of_interest = np.char.upper(["Thy1", "Mbp"])
    result = [(gene_name, res[np.where(gene_names == gene_name)[0]][0]) for gene_name in genes_of_interest]
    print('\n'.join([gene_name + " : " + str(r) for (gene_name, r) in result]))
    return res


@no_grad()
@eval_modules()
def most_expressed_genes(vae, data_loader, M_sampling=100, M_permutation=100000, permutation=False):
    r"""
    :param vae: a VAE model object
    :param data_loader: a ``data_loader`` object
    :param M_sampling: number of samples
    :param M_permutation: M_permutation: 10000 - default value in Romain's code
    :param permutation:
    :return: a table of most expressed gene for each cell. Entry[i][j] represents the cell j's gene_expression
    level for gene i. Gene i is the most expressed gene for cell i.
    """
    gene_names = data_loader.dataset.gene_names
    px_scale, all_labels = de_stats(vae, data_loader, M_sampling)

    most_expressed_genes = []
    results = []

    n_cells = vae.n_labels
    for cell_idx in range(n_cells):
        sample_rate_a = (px_scale[all_labels.view(-1) == cell_idx].view(-1, px_scale.size(1))
                         .cpu().detach().numpy())
        sample_rate_b = (px_scale[all_labels.view(-1) != cell_idx].view(-1, px_scale.size(1))
                         .cpu().detach().numpy())
        samples = np.vstack((sample_rate_a, sample_rate_b))

        list_1 = list(np.arange(sample_rate_a.shape[0]))
        list_2 = list(sample_rate_a.shape[0] + np.arange(sample_rate_b.shape[0]))
        if not permutation:
            u, v = np.random.choice(list_1, size=M_permutation), np.random.choice(list_2, size=M_permutation)
        else:
            u, v = (np.random.choice(list_1 + list_2, size=M_permutation),
                    np.random.choice(list_1 + list_2, size=M_permutation))

        first_set = samples[u]
        second_set = samples[v]
        res = np.mean(first_set >= second_set, 0)
        res = np.log(res + 1e-8) - np.log(1 - res + 1e-8)
        results.append(res)
        most_expressed_gene = gene_names[np.argmax(res)]
        most_expressed_genes.append(most_expressed_gene)

    results = [[res[np.where(gene_names == gene_name)[0]][0] for gene_name in most_expressed_genes] for res in results]
    return most_expressed_genes, results
