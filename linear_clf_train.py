import torch.nn as nn
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
from moco_train import Model
from augmentations import get_augmentations
import torchvision
from torchvision import transforms
import argparse
import os

if __name__ == "__main__":
    # CUDA for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = {'batch_size': 128,
              'shuffle': True,
              'num_workers': 8}

    # Define the parser

    parser = argparse.ArgumentParser(description='get_current_moco')
    parser.add_argument('-moco_name', action="store", dest='moco_name', default="")
    args = parser.parse_args()

    if args.moco_name == "":
        moco_dir = '2021_10_01_14_01'
    else:
        moco_dir = args.moco_name

    #   Path list
    # moco_dir = '2021_09_24_13_50'
    # moco_dir = '2021_09_25_07_02'
    # moco_dir = '2021_10_01_14_01'
    moco_path = r'./results/' + moco_dir
    data_path = r'./imagenette'

    #   Parameters
    max_epochs = 50
    lr = 0.01
    weight_decay = 0.001

    best_acc = -1

    # Generators
    imagenet_augmentation = get_augmentations('imagenet_validation')
    training_set = torchvision.datasets.ImageFolder('imagenette/train', transforms.Compose(imagenet_augmentation))
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = torchvision.datasets.ImageFolder('imagenette/val', transforms.Compose(imagenet_augmentation))
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    #   Create MoCo model
    moco = Model()
    moco.load_state_dict(torch.load(moco_path + '/moco_checkpoint_fq.pt')['model_state_dict'])
    moco.to(device)

    for param in moco.parameters():
        param.requires_grad = False

    #   Create linear model
    linear_clf = nn.Sequential(nn.Linear(2048, 10)).to(device)

    # optimizers
    # optimizer = torch.optim.SGD(linear_clf.parameters(), lr=lr)
    optimizer = torch.optim.Adam(linear_clf.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loss_list, val_loss_list, top_1_acc = [], [], []

    #   log file
    log_path = moco_path + '/clf_log.txt'
    if os.path.exists(log_path):
        os.remove(log_path)
    f = open(log_path, "a")

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        moco.eval()
        avg_loss_train, avg_loss_val, avg_acc_val = [], [], []
        tot_loss, tot_samples, tbar = 0.0, 0, tqdm(training_generator)
        for batch, labels in tbar:
            linear_clf.train()
            optimizer.zero_grad()
            #   Transfer to GPU
            batch, labels = batch.to(device), labels.to(device)

            #   Run through network
            with torch.no_grad():
                output_moco, _ = moco(batch)
            output = linear_clf(output_moco)

            #   Calculate loss
            loss = loss_func(output, labels)
            avg_loss_train.append(loss.item())

            # Adam update: query network
            loss.backward()
            optimizer.step()

            tot_samples += batch.shape[0]
            tot_loss += loss.item() * batch.shape[0]
            tbar.set_description(f'Train Epoch: [{epoch+1}/{max_epochs}] Loss: {tot_loss / tot_samples}')

        acc, tot_samples, vbar = 0.0, 0, tqdm(validation_generator)

        for batch, labels in vbar:
            linear_clf.eval()
            #   Transfer to GPU
            batch, labels = batch.to(device), labels.to(device)

            #   Run through network
            with torch.no_grad():
                output_moco, _ = moco(batch)
                output = linear_clf(output_moco)

            #   Calculate loss
            acc += torch.sum(torch.eq(torch.argmax(output, dim=1), labels))
            tot_samples += batch.shape[0]
            avg_acc_val.append(acc)
            loss = loss_func(output, labels)
            avg_loss_val.append(loss.item())
            vbar.set_description(f'Test Epoch: [{epoch+1}/{max_epochs}] Accuracy: {acc / tot_samples}')

        if acc > acc / tot_samples:
            best_acc = (acc / tot_samples).detach().cpu().numpy()

        epoch_loss = np.mean(avg_loss_train)
        train_loss_list.append(epoch_loss)
        val_loss_list.append(np.mean(avg_loss_val))
        top_1_acc.append(acc / tot_samples)

        status_str = f'Epoch = {epoch + 1}, Loss = {epoch_loss}, Accuracy: {acc / tot_samples}'
        f.write(status_str + '\n')

        torch.save({
            'epoch': epoch,
            'model_state_dict': linear_clf.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, moco_path + '/clf_checkpoint.pt')
    f.write(str(best_acc) + '\n')
    f.close()
