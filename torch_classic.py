import torch
import matplotlib.pyplot as plt
import numpy as np


def k_means(samples: torch.Tensor, num_clusters: int, device: torch.device, mini_batches: int = None,
            iterations: int = 100, tol: float = 1e-4) -> torch.Tensor:
    """
    performs k-means with pytorch
    :param samples: tensor with samples to cluster of shape [N, V] (N is number of samples and V is sample dimension)
    :param num_clusters: number of clusters for k-means
    :param device: pytorch device used
    :param mini_batches: if None use all samples (like a regular kmeans), otherwise each iteration extract a subset.
    see: https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans
    :param iterations: stop algorithm after number of iterations if did not converge
    :param tol: tolerance for convergence
    :return: the centroids of the samples
    """
    indices = torch.randperm(samples.shape[0])[:num_clusters]
    centroids = samples[indices]
    for _ in range(iterations):
        if mini_batches is None:
            subset = samples
        else:
            indices = torch.randperm(samples.shape[0])[:mini_batches]
            subset = samples[indices]
        dists = torch.cdist(subset, centroids).to(device)
        min_dists = torch.min(dists, dim=1, keepdim=True)[0].repeat(1, num_clusters).to(device)
        min_mask = (min_dists == dists).to(device)
        min_mask = torch.unsqueeze(min_mask, dim=2).repeat(1, 1, samples.shape[1]).to(device)
        subset = torch.unsqueeze(subset, dim=1).repeat(1, num_clusters, 1).to(device)
        new_centroids = torch.sum(min_mask * subset, dim=0).to(device) / torch.sum(min_mask, dim=0).to(device)
        if (torch.abs(centroids - new_centroids) <= tol).all():
            # print('converged')
            break
        else:
            centroids = new_centroids
    return centroids


def knn(samples: torch.Tensor, labels: torch.Tensor, k: int, device: torch.device) -> float:
    """
    calculates the accuracy according to KNN. A sample is determined as classified correctly if it at least gets
     k / 2 correct votes (e.g if k is 5 and a sample has 3 close neighbors of the same label its considered to
     be classified correctly).
    :param samples: tensor with samples to classify of shape [N, V] (N is number of samples and V is sample dimension)
    :param labels: tensor with samples real labels of shape [N, 1]
    :param k: number of neighbors to consider
    :param device: pytorch device used
    :return:
    """
    dists = torch.cdist(samples, samples).to(device)
    # print(f'dists shape: {dists.shape}')
    # filter out the distances of samples from themselves:
    dists = dists + torch.diag(torch.ones(samples.shape[0]) * float('inf')).to(device)

    topk = torch.topk(dists, k=k, dim=1, largest=False)
    # print(f'topk[1] shape: {topk[1].shape}')
    votes = torch.take(labels, topk[1].to(device)).to(device)
    # print(f'votes shape: {votes.shape}')
    labels = labels.repeat((1, k)).to(device)
    # print(f'labels shape: {labels.shape}')
    correct_votes = (votes == labels).type(torch.IntTensor).to(device)
    num_correct_votes = torch.sum(correct_votes, dim=1).to(device)
    correct_classifications = (num_correct_votes >= (k / 2)).type(torch.DoubleTensor).to(device)
    accuracy = torch.sum(correct_classifications) / samples.shape[0]
    return accuracy.item()


if __name__ == '__main__':
    # k-means sanity check
    inp = torch.randn(1000, 2)
    num_clusters = 10
    centroids = k_means(inp, num_clusters, torch.device('cpu'), mini_batches=None, iterations=100)
    inp = inp.detach().cpu().numpy()
    centroids = centroids.detach().cpu().numpy()
    plt.scatter(np.concatenate([inp[:, 0], centroids[:, 0]]),
                np.concatenate([inp[:, 1], centroids[:, 1]]),
                c=np.concatenate([np.ones(inp.shape[0]), np.ones(num_clusters) * 2]))
    plt.title('k-means sanity check')
    plt.show()

    # knn sanity check
    inp = torch.Tensor([[0, 0], [0.5, 0], [-0.5, 0],
                        [0, 10], [0.5, 10], [-0.5, 10],
                        [0, 20], [0.5, 20], [-0.5, 20]])
    labels1 = torch.Tensor([[0], [1], [2],
                            [0], [1], [2],
                            [0], [1], [2]])
    labels2 = torch.Tensor([[0], [0], [0],
                            [1], [1], [1],
                            [2], [2], [2]])
    print(f'accuracy with labels1 tensor: {knn(inp, labels1, k=3, device=torch.device("cpu"))}')
    print(f'accuracy with labels2 tensor: {knn(inp, labels2, k=3, device=torch.device("cpu"))}')
