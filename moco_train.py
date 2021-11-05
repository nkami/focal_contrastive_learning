import torch
import torchvision.models as models
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from augmentations import get_augmentations, TwoCropsTransform
from torchvision import transforms
from torch_classic import k_means, knn
from pathlib import Path
from typing import Tuple
#from loss_info import load_model

"""
max_epochs: number of epochs to train
q_size: queue size
contrastive_momentum: the contrastive momentum used
temperature: temperature used
lr: learning rate
SGD_momentum: momentum for sgd optimizer
weight_decay: weight decay for optimizer
batch_size: training batch_size
training_augmentations: the augmentations used. can be moco_v1, moco_v2 (the augmentations used for first and second 
moco respectively). Can also be imagenet_validation as the augmentations used for validation.
knn_test_freq: The frequency of performing a knn test during training. each knn_test_freq epochs a knn test will start, 
this test can give an indication on the quality of the current embeddings of the model.
k: k used for the knn test.
gamma: gamma used for focal loss.
filter_logits_starting_epoch: the epoch to start filtering from (e.g., if the current training epoch is bigger than 
filter_logits_starting_epoch filtering will be applied).
clustered_logits_starting_epoch: the epoch to start clustering from (e.g., if the current training epoch is bigger than 
clustered_logits_starting_epoch clustering will be applied).
update_centroids_freq: the frequency of updating the centroids of the queue for filtering / clustering
num_centroids: number of centroids to use for filtering / clustering
k_means_mini_batches: if filtering is too time consuming you can use a smaller batch of the queue. otherwise, set to 
None (if the queue is relatively small <= 2 ** 15 just set it to None).
pretrained: load a pretrained backbone from a previous run (see line 185).
num_workers: number of worker in the dataloader
"""
_HYPER_PARAMS = {'max_epochs': 1250,
                 'q_size': int(2 ** 12),
                 'contrast_momentum': 0.999,
                 'temperature': 0.07,
                 'lr': 0.001,
                 'SGD_momentum': 0.9,
                 'weight_decay': 0.0001,
                 'batch_size': 90, #192,
                 'training_augmentations': 'moco_v2',
                 'knn_test_freq': 200,
                 'k': 5,
                 'gamma': 0,
                 'filter_logits_starting_epoch': -1,
                 'clustered_logits_starting_epoch': 1500,
                 'update_centroids_freq': 1,
                 'num_centroids': 20,
                 'k_means_mini_batches': None,
                 'pretrained': True,
                 'num_workers': 7,
                 }


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()
        self.feature_extractor = models.resnet50(pretrained=False)
        self.feature_extractor.fc = nn.Identity()

        # MLP head
        self.mlp_head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, feature_dim))

    def forward(self, x):
        x = self.feature_extractor(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.mlp_head(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


def _get_soft_labels(samples: torch.Tensor, centroids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    infer the labels of all the samples according to the centroids locations.
    :param samples: the queue used for moco
    :param centroids: the current centroids calculated with k-means
    :return: soft labels of all queue samples in shape (N, 1)
    """
    dists = torch.cdist(samples, centroids).to(device)
    min_indices = torch.min(dists, dim=1, keepdim=True)[1].to(device)
    return min_indices.reshape((-1, 1)).to(device)


def _filter_logits(neg_logits: torch.Tensor, batch_labels: torch.Tensor, queue_labels: torch.Tensor,
                   device: torch.device) -> torch.Tensor:
    """
    make the product of images in the queue of the same class as the queries very negative.
    essentially these images wont affect the loss or gradients.
    :param neg_logits: the negative logits of current batch. tensor of shape [N, Q] where N is batch size
    and Q is queue size.
    :param batch_labels: the labels of the samples in current batch. tensor of shape [N, 1] where N is batch size.
    :param queue_labels: the labels of samples in queue. tensor of shape [Q, 1] where Q is queue size.
    :return: filtered logits
    """
    queue_labels = queue_labels.repeat(1, neg_logits.shape[0]).to(device)
    batch_labels = batch_labels.repeat(1, neg_logits.shape[1]).to(device)
    mask = (batch_labels == queue_labels.T).to(device)
    l_neg[mask] = float('-inf')
    # it seems manually assigning the value also zeroes out the relevant gradients. the following code supports this
    # assumption:
    #     x = torch.Tensor([1, 1])
    #     x.requires_grad = True
    #     y = 5 * (x ** 2)
    #     y[0] = float('-inf')
    #     y.retain_grad()
    #     y.backward(torch.Tensor([1, 1]))
    #     print(x.grad)
    #     print(y.grad)
    return l_neg


def _get_features_and_labels(model: nn.Module, device: torch.device,
                             dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns the features and true labels of the validation data set
    :param model: moco encoder
    :param device: pytorch device
    :param dataloader: validation set data loader
    :return: features tensor and labels tensor
    """
    tbar = tqdm(dataloader, desc='Feature extracting')
    features, labels = None, None
    model.eval()
    with torch.no_grad():
        for batch, batch_labels in tbar:
            batch, batch_labels = batch.to(device), batch_labels.to(device)
            _, batch_features = model(batch)
            if features is None:
                features = batch_features
                labels = batch_labels
            else:
                features = torch.cat((features, batch_features))
                labels = torch.cat((labels, batch_labels))
    return features, labels.reshape((-1, 1))


if __name__ == "__main__":
    # CUDA for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #   Path list
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M")
    name = 'gamma_' + str(_HYPER_PARAMS['gamma'])+'_'
    if _HYPER_PARAMS['pretrained']:
        name = name + 'pretrained_'
    if _HYPER_PARAMS['filter_logits_starting_epoch'] <= 0:
        name = name + 'filtered_'
    if _HYPER_PARAMS['clustered_logits_starting_epoch'] <= 0:
        name = name + 'combined_'
    if _HYPER_PARAMS['lr'] < 0.001:
        name = name + 'lr_' + str(_HYPER_PARAMS['lr'])+'_E800_'
    if _HYPER_PARAMS['num_centroids'] != 10:
        name = name + 'k'+str(_HYPER_PARAMS['num_centroids'])+'_'


    res_path = r'./results/moco_' + name + current_time
    Path(res_path).mkdir(parents=True, exist_ok=True)

    # Generators
    training_augmentations = get_augmentations(_HYPER_PARAMS['training_augmentations'])
    training_set = torchvision.datasets.ImageFolder('imagenette/train',
                                                    TwoCropsTransform(transforms.Compose(training_augmentations)))
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=_HYPER_PARAMS['batch_size'],
                                                     shuffle=True, num_workers = _HYPER_PARAMS['num_workers'])

    validation_augmentations = get_augmentations('imagenet_validation')
    validation_set = torchvision.datasets.ImageFolder('imagenette/val', transforms.Compose(validation_augmentations))

    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=_HYPER_PARAMS['batch_size'],
                                                       shuffle=False, num_workers = _HYPER_PARAMS['num_workers'])

    # Create f_q and f_k
    if _HYPER_PARAMS['pretrained']:
        #f_q = LoadedModel().to(device=device)
        f_q = Model()
        f_q.load_state_dict(torch.load('./results/moco_gamma_0_2021_10_13_11_07/moco_checkpoint_fq.pt')['model_state_dict'])
        f_q = f_q.to(device)
    else:
        f_q = Model().to(device=device)
        
    f_k = copy.deepcopy(f_q).to(device)
    for param in f_k.parameters():
        param.requires_grad = False

    # optimizers
    optimizer = torch.optim.SGD(f_q.parameters(), lr=_HYPER_PARAMS['lr'],
                                momentum=_HYPER_PARAMS['SGD_momentum'], weight_decay=_HYPER_PARAMS['weight_decay'])

    loss_list = []

    #   initialize queue
    queue = F.normalize(torch.randn(128, _HYPER_PARAMS['q_size']), dim=0).to(device)
    # print(f'starting queue shape: {queue.shape}')
    queue_centroids = None

    #   log file
    f = open(res_path + '/moco_log.txt', "a")
    f.write(f'{_HYPER_PARAMS}\n')

    for epoch in range(_HYPER_PARAMS['max_epochs']):
        # Training
        f_q.train()
        f_k.train()
        avg_loss = []
        tot_loss, tot_samples, tbar = 0.0, 0, tqdm(training_generator)
        for iter, (batch, _) in enumerate(tbar):
            # training generator produces [[aug1_batch, aug2_batch], label_tensor]
            optimizer.zero_grad()

            # Transfer to GPU
            q_batch, k_batch = batch[0].to(device), batch[1].to(device)

            #   Run through models
            _, q_output_batch = f_q(q_batch.type(torch.float))
            _, k_output_batch = f_k(k_batch.type(torch.float))

            k_output_batch = k_output_batch.detach()

            #   Create logits
            b_size = k_output_batch.shape[0]
            f_size = k_output_batch.shape[1]

            l_pos = torch.bmm(q_output_batch.view(b_size, 1, f_size), k_output_batch.view(b_size, f_size, 1))
            l_neg = torch.mm(q_output_batch.view(b_size, f_size), queue)

            # filter negative logits according to labels obtained from k-means
            if _HYPER_PARAMS['filter_logits_starting_epoch'] <= epoch+1:
                if iter % _HYPER_PARAMS['update_centroids_freq'] == 0 or queue_centroids is None:
                    queue_centroids = k_means(queue.T, _HYPER_PARAMS['num_centroids'], device,
                                              mini_batches=None)
                # print(f'queue shape: {queue.shape}, queue_centroids shape: {queue_centroids.shape}')
                queue_soft_labels = _get_soft_labels(queue.T, queue_centroids, device)
                batch_soft_labels = _get_soft_labels(q_output_batch, queue_centroids, device)

                l_neg = _filter_logits(l_neg, batch_soft_labels, queue_soft_labels, device)
            
            p = []
            if _HYPER_PARAMS['clustered_logits_starting_epoch'] <= epoch+1:
                queue_centroids = k_means(queue.T, _HYPER_PARAMS['num_centroids'], device,
                                              mini_batches=_HYPER_PARAMS['k_means_mini_batches'])
                # print(f'queue shape: {queue.shape}, queue_centroids shape: {queue_centroids.shape}')
                queue_soft_labels = _get_soft_labels(queue.T, queue_centroids, device)
                batch_soft_labels = _get_soft_labels(q_output_batch, queue_centroids, device)

                mask = queue_soft_labels.unsqueeze(0).repeat(batch_soft_labels.shape[0], 1 , 1)
                mask = mask == batch_soft_labels.unsqueeze(2)
                zeros = torch.zeros((mask.shape[0], 1), dtype= torch.bool, device  = device)
                mask = torch.cat([zeros, mask.squeeze()], dim = 1)

                logits = torch.cat((l_pos.view(-1, 1), l_neg), dim=1)
                logits_softmax = (logits / _HYPER_PARAMS['temperature']).softmax(dim = 1)
                logits_softmax_indexes = torch.arange(logits_softmax.shape[0], device = device).unsqueeze(1).repeat(1, logits_softmax.shape[1])

                logits_softmax_masked = logits_softmax[mask]
                logits_softmax_index_masked = logits_softmax_indexes[mask]
                p = logits_softmax[:, 0]
                p.scatter_add(0, logits_softmax_index_masked, logits_softmax_masked).unsqueeze(1)
                log_p = -torch.log(p)

            else:
                logits = torch.cat([l_pos.view(-1, 1), l_neg], dim=1)
                logits = logits / _HYPER_PARAMS['temperature']

                # contrastive focal loss
                labels = torch.zeros(b_size, dtype=torch.int64).to(device)

                log_p = torch.nn.functional.cross_entropy(logits, labels, reduction='none')

            if len(p) == 0:
                p = torch.exp(-1 * log_p)

            loss = torch.mean(((1 - p) ** _HYPER_PARAMS['gamma']) * log_p)
            avg_loss.append(loss.item())

            # Optimizer update: query network
            loss.backward()
            optimizer.step()

            # Momentum update: key network
            for q_parameters, k_parameters in zip(f_q.parameters(), f_k.parameters()):
                k_parameters.data = (k_parameters.data * _HYPER_PARAMS["contrast_momentum"] +
                                     q_parameters.data * (1. - _HYPER_PARAMS["contrast_momentum"]))

            queue = torch.cat([queue, k_output_batch.T], dim=1).to(device)
            queue = queue[:, k_output_batch.T.shape[1]:]

            # Save status
            status_str = "Epoch = " + str(epoch+1) + ", Iteration = " + str(iter) + ', Loss = ' + str(loss.item())
            f.write(status_str + '\n')

            tot_samples += k_output_batch.shape[0]
            tot_loss += loss.item() * k_output_batch.shape[0]
            tbar.set_description(f'Train Epoch: [{epoch+1}/{_HYPER_PARAMS["max_epochs"]}] Loss: '
                                 f'{tot_loss / tot_samples}')

        # do a knn test
        if (epoch+1) % _HYPER_PARAMS['knn_test_freq'] == 0:
            features, labels = _get_features_and_labels(f_q, device, validation_generator)
            acc = knn(features, labels, k=_HYPER_PARAMS['k'], device=device)
            f.write(f'knn accuracy: {acc} at epoch: {epoch+1}\n')
            pbar = tqdm(['_'])
            for _ in pbar:
                pbar.set_description(f'KNN accuracy in epoch {epoch+1} is: {acc}')

        epoch_loss = np.mean(avg_loss)
        loss_list.append(epoch_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': f_q.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, res_path + '/moco_checkpoint_fq.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': f_k.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, res_path + '/moco_checkpoint_fk.pt')

    f.close()
    plt.figure()
    plt.plot(np.array(loss_list))
    plt.title('MoCo Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(res_path + '/moco_loss_graph.png')
