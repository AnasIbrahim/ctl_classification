import argparse, random, copy
import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

from dataset_armbench import ArmBenchDataset, test_armbench

#class CTLModel(nn.Module):
#    def __init__(self):
#        super(CTLModel, self).__init__()
#        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
#
#    def forward(self, img):
#        output = self.resnet(img)
#        output = output.view(output.size()[0], -1)
#        return output


class CentroidTripletLoss(nn.Module):
    def __init__(self):
        super(CentroidTripletLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        # get centroids of anchor, positive and negative
        anchor_centroid = torch.mean(anchor, dim=0)
        positive_centroid = torch.mean(positive, dim=0)
        negative_centroid = torch.mean(negative, dim=0)

        # calculate distances between anchor and centroids
        anchor_positive_distance = torch.dist(anchor_centroid, positive_centroid)
        anchor_negative_distance = torch.dist(anchor_centroid, negative_centroid)

        # calculate loss
        loss = torch.clamp(anchor_positive_distance - anchor_negative_distance + 0.1, min=0.0)
        return loss


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for p in model.module.parameters():
        p.requires_grad = True

    criterion = CentroidTripletLoss()

    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_output, positive_output, negative_output = model(anchor), model(positive), model(negative)
        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        log_interval = 50
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader.dataset), loss.item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train backbones for object classification (reidentification) using centroid loss using ArmBench dataset')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (= number of objects used in one training step) (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (= number of objects used in one testing step) (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1 for reproducibility)')
    parser.add_argument('--save-path', action='store_true', default=False,
                        help='For Saving the log and the models')
    args = parser.parse_args()

    armbench_dataset_path = '/media/gouda/ssd_data/datasets/armbench-object-id-0.1'
    output_path = '/home/gouda/segmentation/ctl_training_output/gouda/train_1'
    
    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    cuda_kwargs = {'num_workers': 4,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_dataset = ArmBenchDataset(armbench_dataset_path, mode='train')
    test_dataset = ArmBenchDataset(armbench_dataset_path, mode='test')
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("Loaded training and test dataset")

    #model = CTLModel().to(device)
    #model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    model = model.to(device)
    model = nn.DataParallel(model)
    print("Model created")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Run test before training
    test_armbench(model, device, test_loader, 0)

    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_armbench(model, device, test_loader, epoch)
        scheduler.step()

        print("Epoch Time:")
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()

        torch.save(model.state_dict(), os.path.join(output_path, 'models', 'epoch-'+ str(epoch) + ".pt"))


if __name__ == '__main__':
    main()
