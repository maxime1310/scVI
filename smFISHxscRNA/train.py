from itertools import cycle

import torch

from smFISHxscRNA.utils import to_cuda, enable_grad


@enable_grad()
def train_FISHVAE_jointly(vae, data_loader_train,  data_loader_train_fish,
                  n_epochs=20, lr=0.001, kl=None, weight_decay=0.25, benchmark=False):
    # Defining the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vae.parameters()), lr=lr, eps=0.01, weight_decay=weight_decay)

    # Training the model
    for epoch in range(n_epochs):

        for i_batch, (tensors_train, tensors_train_fish) in enumerate(zip(data_loader_train, cycle(data_loader_train_fish))):

            if False:
                tensors_train = to_cuda(tensors_train)
                tensors_train_fish = to_cuda(tensors_train_fish)

            sample_batch_train, local_l_mean_train, local_l_var_train, batch_index_train, labels_train = tensors_train
            sample_batch_train = sample_batch_train.type(torch.float32)
            sample_batch_train_fish, local_l_mean_train_fish, local_l_var_train_fish, batch_index_train_fish, labels_train_fish, _, _ = tensors_train_fish
            sample_batch_train_fish = sample_batch_train_fish.type(torch.float32)
            if kl is None:
                kl_ponderation = min(1, epoch /2000.)
            else:
                kl_ponderation = kl
            batch_index_train = torch.zeros_like(local_l_mean_train)
            reconst_loss_train, kl_divergence_train = vae(sample_batch_train, local_l_mean_train, local_l_var_train,
                                                          batch_index=batch_index_train, y=labels_train, mode="scRNA")

            train_loss = torch.mean(reconst_loss_train + kl_ponderation * kl_divergence_train)
            batch_index_train_fish = torch.ones_like(local_l_mean_train_fish)
            reconst_loss_train_fish, kl_divergence_train_fish = vae(sample_batch_train_fish,
                                                                                       local_l_mean_train_fish,
                                                                                       local_l_var_train_fish,
                                                                                       batch_index=batch_index_train_fish,
                                                                                       y=labels_train_fish, mode="smFISH")

            train_loss_fish = torch.mean(reconst_loss_train_fish + kl_ponderation * kl_divergence_train_fish)
            if vae.weight == True:
                train_loss = train_loss * sample_batch_train.size(0) + train_loss_fish * sample_batch_train_fish.size(0) * vae.n_input/ vae.n_input_fish
            if vae.weight == False:
                train_loss = train_loss * sample_batch_train.size(0) + train_loss_fish * sample_batch_train_fish.size(0)

            train_loss /= (sample_batch_train.size(0) + sample_batch_train_fish.size(0))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


