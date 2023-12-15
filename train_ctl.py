import argparse, random, copy
import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

from dataset_armbench import ArmBenchDataset, test_armbench
from dataset_bop import BOPDataset, test_bop


class CTLModel(nn.Module):
    def __init__(self):
        super(CTLModel, self).__init__()

        # ResNet
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        # remove last 2 layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # VIT
        #self.backbone = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)

    def forward(self, img):
        output = self.backbone(img)
        output = output.view(output.size()[0], -1)
        return output


#class CentroidTripletLoss(nn.Module):
#    def __init__(self):
#        super(CentroidTripletLoss, self).__init__()
#
#    def forward(self, anchor, positive, negative):
#        # get centroids of anchor, positive and negative
#        anchor_centroid = torch.mean(anchor, dim=0)
#        positive_centroid = torch.mean(positive, dim=0)
#        negative_centroid = torch.mean(negative, dim=0)
#
#        # calculate distances between anchor and centroids
#        anchor_positive_distance = torch.nn.functional.cosine_similarity(anchor_centroid, positive_centroid, dim=0)
#        anchor_negative_distance = torch.nn.functional.cosine_similarity(anchor_centroid, negative_centroid, dim=0)
#
#        # calculate loss
#        #loss = torch.clamp(anchor_positive_distance - anchor_negative_distance + 0.1, min=0.0)
#        loss = torch.mean((anchor_positive_distance - anchor_negative_distance) ** 2)
#        return loss


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for p in model.module.parameters():
        p.requires_grad = True

    #criterion = CentroidTripletLoss()
    criterion = nn.TripletMarginLoss(margin=0.1, p=2)

    for batch_idx, batch_imgs in enumerate(train_loader):
        # run the model
        optimizer.zero_grad()
        all_embeddings = model(batch_imgs.to(device))
        # normalize all embeddings
        # TODO check if this is needed and if query and gallery embeddings should be normalized separately?
        all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=0, p=2)
        # reshape tensor to (X,6,3,256,256) to be able to calculate centroids per object
        # view must be used instead of reshape
        feat_dim = all_embeddings.shape[-1]
        all_embeddings = all_embeddings.view((-1, 6, feat_dim))
        # calculate centroids per object
        # TODO should EmbeddingBag be used instead of mean?
        all_centroids = torch.mean(all_embeddings, dim=1)
        # reshape tensor to (X,3,3,256,256)
        # view must be used instead of reshape
        all_centroids = all_centroids.view((-1, 3, feat_dim))
        # separate anchor, positive and negative embeddings
        anchor_centroids = all_centroids[:, 0]
        positive_centroids = all_centroids[:, 1]
        negative_centroids = all_centroids[:, 2]
        # calculate loss
        loss = criterion(anchor_centroids, positive_centroids, negative_centroids)
        loss.backward()

        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch,
            batch_idx, int(len(train_loader.dataset)/args.train_batch_size),
            100. * batch_idx / int(len(train_loader.dataset) / args.train_batch_size), loss.item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train backbones for object classification (reidentification) using centroid loss using ArmBench dataset')
    parser.add_argument('--train-batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (= number of objects used in one training step) (default: 64)')
    parser.add_argument('--armbench-test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (= number of objects used in one testing step)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1 for reproducibility)')
    parser.add_argument('--save-path', action='store_true', default=False,
                        help='For Saving the log and the models')
    args = parser.parse_args()

    armbench_dataset_path = '/media/gouda/ssd_data/datasets/armbench-object-id-0.1'
    hope_dataset_path = '/media/gouda/ssd_data/datasets/hope/classification'
    output_path = '/home/gouda/segmentation/ctl_training_output/gouda/train_1'
    
    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    train_kwargs = {'batch_size': args.train_batch_size,
                    'shuffle': False}  # TODO change to True
    test_kwargs = {'batch_size': args.armbench_test_batch_size,
                   'shuffle': False}
    cuda_kwargs = {'num_workers': 4,
                   'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    armbench_train_dataset = ArmBenchDataset(armbench_dataset_path, mode='train')
    armbench_test_dataset = ArmBenchDataset(armbench_dataset_path, mode='test')
    bop_test_dataset = BOPDataset(hope_dataset_path)
    armbench_train_loader = torch.utils.data.DataLoader(armbench_train_dataset, collate_fn=ArmBenchDataset.train_collate_fn, **train_kwargs)
    armbench_test_loader = torch.utils.data.DataLoader(armbench_test_dataset, collate_fn=ArmBenchDataset.test_collate_fn, **test_kwargs)
    bop_test_loader = torch.utils.data.DataLoader(bop_test_dataset, **test_kwargs)
    print("Loaded training and test dataset")

    model = CTLModel().to(device)
    #model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    #model = model.to(device)

    model = nn.DataParallel(model)
    print("Model created")

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Run test before training
    test_bop(model, device, bop_test_loader, batch_size=150, epoch=0)
    test_armbench(model, device, armbench_test_loader, args.armbench_test_batch_size, 0)

    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, armbench_train_loader, optimizer, epoch)
        test_bop(model, device, bop_test_loader, batch_size=150, epoch=0)
        test_armbench(model, device, armbench_test_loader, args.armbench_test_batch_size, 0)
        scheduler.step()

        print("Epoch Time:")
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()

        torch.save(model.state_dict(), os.path.join(output_path, 'models', 'epoch-'+ str(epoch) + ".pt"))


if __name__ == '__main__':
    main()
