import torchvision
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse
from argparse import Namespace
import numpy as np
from augmentations import get_augmentations, TwoCropsTransform
from torchvision import transforms
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from moco_train import Model


def load_model(checkpoint: str) -> nn.Module:
    # """
    # loading checkpoints of moco versions from the official github repo: https://github.com/facebookresearch/moco
    # (the tar extension is fake, after downloading just remove it).
    # :param checkpoint: can be 'moco_v1_200ep_pretrain.pth', 'moco_v2_200ep_pretrain.pth' or
    #                     'moco_v2_800ep_pretrain.pth'
    # :return: resnet-50 backbone for moco
    # """
    # model = torchvision.models.resnet50(num_classes=128)
    # state_dict = torch.load(checkpoint)
    # modified_dict = {key.replace('module.encoder_q.', ''): val for key, val in state_dict['state_dict'].items()}

    # if checkpoint == 'moco_v1_200ep_pretrain.pth':
    #     model.load_state_dict(modified_dict)
    # else:
    #     model.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), model.fc)
    #     model.load_state_dict(modified_dict)

    model = Model()
    model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    model.eval()
    return model


def _get_loss_samples(model: nn.Module, dataloader: torch.utils.data.DataLoader, queue_size: int,
                      temp: float, num_samples: int, unique_key: bool, combined_key: bool) -> np.ndarray:
    """
    :param model: trained moco encoder
    :param dataloader: dataloader of samples
    :param queue_size: size of queue to use for calculating the negative logits
    :param temp: temperature used for loss function
    :param num_samples: number of loss samples to get
    :param unique_key: if to filter out images in the queue of the same class as the queries
    :return: loss samples
    """
    queue, queue_labels = None, None
    loss_samples = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for count, sample in enumerate(dataloader):
            print("1 - "+str(int(100*count/(2*len(dataloader))))+'%', flush = True)
            # sample is [[aug1_batch, aug2_batch], label_tensor]
            x1 = sample[0][0].to(device)
            #queries = model(x1)
            _, queries = model(x1)
            queries = nn.functional.normalize(queries, dim=1)
            if queue is None:
                queue = queries
                queue_labels = sample[1]
            else:
                queue = torch.cat((queue, queries))
                queue_labels = torch.cat((queue_labels, sample[1]))
            if queue.shape[0] > queue_size:
                queue = queue[:queue_size, :]
                queue_labels = queue_labels[:queue_size]
                queue_labels = queue_labels.repeat(x1.shape[0], 1)
                break
        queue = queue.to(device)
        for count, sample in enumerate(dataloader):
            print("2 - "+str(50+int(100*count/(2*len(dataloader))))+'%', flush = True)

            x1, x2 = sample[0][0].to(device), sample[0][1].to(device)
            #queries = model(x1)
            _,queries = model(x1)
            queries = nn.functional.normalize(queries, dim=1)
            keys = model(x2)
            _, keys = model(x2)

            keys = nn.functional.normalize(keys, dim=1)
            l_pos = torch.bmm(queries.view((-1, 1, 128)), keys.view(-1, 128, 1)).reshape((-1, 1)).to(device)
            l_neg = queries @ queue.T

            if unique_key:
                # make the product of images in the queue of the same class as the queries very negative.
                # essentially these images wont affect the loss
                # TODO: what about the normalization beforehand in the queue (when you normalize you take
                #  into account the -inf logits)? -> actually it doesnt seem to be an issue because the normalization
                #  is on dim=1

                # cur_labels = sample[1].reshape((-1, 1)).repeat((1, queue_size))
                # mask = cur_labels == queue_labels
                # l_neg[mask] = float('-inf')
                num_centroids = 10
                k_means_mini_batches = None
                from moco_train import _get_soft_labels
                from torch_classic import k_means, knn
                queue_centroids = k_means(queue, num_centroids, device,mini_batches=None)
                queue_soft_labels = _get_soft_labels(queue, queue_centroids, device)
                batch_soft_labels = _get_soft_labels(queries, queue_centroids, device)
                queue_labels = queue_soft_labels.repeat(1, l_neg.shape[0]).to(device)
                batch_labels = batch_soft_labels.repeat(1, l_neg.shape[1]).to(device)
                mask = (batch_labels == queue_labels.T).to(device)
                l_neg[mask] = float('-inf')
                logits = torch.cat((l_pos, l_neg), dim=1)
                labels = torch.zeros(queries.shape[0]).to(device)
                cur_loss = nn.functional.cross_entropy((logits / temp).type(torch.float), labels.type(torch.long),
                                                    reduction='none')
            elif combined_key:
                num_centroids = 10
                k_means_mini_batches = None
                from moco_train import _get_soft_labels
                from torch_classic import k_means, knn
                queue_centroids = k_means(queue, num_centroids, device,mini_batches=k_means_mini_batches)
                queue_soft_labels = _get_soft_labels(queue, queue_centroids, device)
                batch_soft_labels = _get_soft_labels(queries, queue_centroids, device)

                mask = queue_soft_labels.unsqueeze(0).repeat(batch_soft_labels.shape[0], 1 , 1)
                mask = mask == batch_soft_labels.unsqueeze(2)
                ones = torch.ones((mask.shape[0], 1), dtype= torch.bool, device  = device)
                mask = torch.cat([ones, mask.squeeze()], dim = 1)

                logits = torch.cat((l_pos, l_neg), dim=1)
                logits_softmax = (logits / temp).softmax(dim = 1)
                new_pos_softmax = (mask * logits_softmax).sum(axis = 1)
                cur_loss = -torch.log(new_pos_softmax)

            else:
                logits = torch.cat((l_pos, l_neg), dim=1)
                labels = torch.zeros(queries.shape[0]).to(device)
                cur_loss = nn.functional.cross_entropy((logits / temp).type(torch.float), labels.type(torch.long),
                                                    reduction='none')
            if loss_samples is None:
                loss_samples = cur_loss
            else:
                loss_samples = torch.cat((loss_samples, cur_loss))
            if loss_samples.shape[0] > num_samples:
                loss_samples = loss_samples[:num_samples]
                break
    return loss_samples.detach().cpu().numpy()


def _plot_loss_dist(unfiltered_loss_samples: np.ndarray, filtered_loss_samples: np.ndarray) -> None:
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].hist(unfiltered_loss_samples, bins=200)
    ax[0].set_xlabel('Loss')
    ax[0].set_ylabel('Samples')
    ax[0].set_title('Unfiltered Loss Distribution')

    ax[1].hist(filtered_loss_samples, bins=200)
    ax[1].set_xlabel('Loss')
    ax[1].set_ylabel('Samples')
    ax[1].set_title('Filtered Loss Distribution')

    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M")
    Path('loss_info_output').mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(f'loss_info_output/loss_dist_{current_time}.png', dpi=100)


def _get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    #parser.add_argument('--ckpt', default='./facebook_moco_checkpoints/moco_v2_800ep_pretrain.pth')
    
    path = './results/moco_gamma_0_2021_10_13_11_07/moco_checkpoint_fq.pt'
    parser.add_argument('--ckpt', default=path)
    parser.add_argument('--qs', default=2 ** 13, type=int, help='queue size')
    parser.add_argument('--s', default=5000, type=int, help='number of loss samples')
    parser.add_argument('--t', default=0.07, type=float, help='temperature')
    args = parser.parse_args()
    return args


def _main() -> None:
    """
    show loss distribution with moco model
    :return:
    """
    num_workers = 8
    args = _get_args()
    model = load_model(args.ckpt)
    selected_augmentations = 'moco_v1' if 'moco_v1' in args.ckpt else 'moco_v2'
    augmentation = get_augmentations(selected_augmentations)
    ds = torchvision.datasets.ImageFolder('imagenette/train', TwoCropsTransform(transforms.Compose(augmentation)))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers = num_workers)
    print('Extracting loss samples...')

    unfiltered_loss_samples = _get_loss_samples(model, dl, args.qs, args.t, args.s, False, False)
    filtered_loss_samples = _get_loss_samples(model, dl, args.qs, args.t, args.s, True, False)

    print('Done!')
    _plot_loss_dist(unfiltered_loss_samples, filtered_loss_samples)


if __name__ == '__main__':
    _main()
