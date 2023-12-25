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
from torch.utils.tensorboard import SummaryWriter

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


def train(model, device, train_loader, optimizer, writer, epoch):
    print("Training epoch: " + str(epoch))
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    criterion = nn.TripletMarginLoss(margin=0.1, p=2)

    for batch_idx, batch_imgs in enumerate(tqdm.tqdm(train_loader)):
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

        # log loss every 50 batches
        if batch_idx % 50 == 0:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #    epoch,
        #    batch_idx, int(len(train_loader.dataset)/args.train_batch_size),
        #    100. * batch_idx / int(len(train_loader.dataset) / args.train_batch_size), loss.item()))


def main():
    armbench_train_batch_size = 40
    armbench_test_batch_size = 20
    bop_test_batch_size = 500
    epochs = 15
    #seed = 1

    armbench_dataset_path = '/raid/datasets/armbench-object-id-0.1'
    armbench_processed_data_file_train = '/home/gouda/segmentation/ctl_classification/processed_armbench_dataset_train.pickle'
    armbench_processed_data_file_test =  '/home/gouda/segmentation/ctl_classification/processed_armbench_dataset_test.pickle'
    hope_dataset_path = '/raid/datasets/hope/classification'
    output_path = '/home/gouda/segmentation/scratch_training_output/train_1'
    
    train_kwargs = {'batch_size': armbench_train_batch_size,
                    'shuffle': True}
    test_kwargs = {'batch_size': armbench_test_batch_size,
                   'shuffle': False}
    cuda_kwargs = {'num_workers': 8,
                   'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    lr = 0.001
    step_size = 5
    gamma = 0.1

    #torch.manual_seed(seed)
    device = torch.device("cuda")

    if not os.path.exists(os.path.join(output_path, 'models')):
        os.makedirs(os.path.join(output_path, 'models'))

    # uncomment next paragraph to save processed data
    #armbench_train_dataset = ArmBenchDataset(armbench_dataset_path, mode='train', portion=None, saved_processed_data_file=None)
    #armbench_test_dataset = ArmBenchDataset(armbench_dataset_path, mode='test', portion=None, saved_processed_data_file=None)
    #armbench_train_dataset.save_processed_data(armbench_processed_data_file_train)
    #armbench_test_dataset.save_processed_data(armbench_processed_data_file_test)
    #exit()

    armbench_train_dataset = ArmBenchDataset(armbench_dataset_path, mode='train', portion=None, saved_processed_data_file=armbench_processed_data_file_train)
    armbench_test_dataset = ArmBenchDataset(armbench_dataset_path, mode='test', portion=None, saved_processed_data_file=armbench_processed_data_file_test)

    bop_test_dataset = BOPDataset(hope_dataset_path)
    armbench_train_loader = torch.utils.data.DataLoader(armbench_train_dataset, collate_fn=ArmBenchDataset.train_collate_fn, **train_kwargs)
    armbench_test_loader = torch.utils.data.DataLoader(armbench_test_dataset, collate_fn=ArmBenchDataset.test_collate_fn, **test_kwargs)
    bop_test_loader = torch.utils.data.DataLoader(bop_test_dataset, **test_kwargs)
    print("Loaded training and test dataset")

    #model = CTLModel().to(device)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    #model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    #model = torchvision.models.vit_l_32(weights=torchvision.models.ViT_L_32_Weights.DEFAULT)
    model = model.to(device)

    model_train_opt = torch.compile(model)

    #model = nn.DataParallel(model)

    print("Model created")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler lowers LR by a factor of 10 every 3 epochs
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    writer = SummaryWriter(os.path.join(output_path, 'logs'))

    # Run test before training
    print("=====================")
    print("Running test before training")
    test_bop(device, model, bop_test_loader, bop_test_batch_size, writer, epoch=0)
    test_armbench(device, model, armbench_test_loader, writer, epoch=0)

    start_time = datetime.datetime.now()

    for epoch in range(1, epochs + 1):
        print("=====================")
        print("Epoch: " + str(epoch))
        train(model_train_opt, device, armbench_train_loader, optimizer, writer, epoch)
        test_bop(device, model, bop_test_loader, bop_test_batch_size, writer, epoch)
        test_armbench(device, model, armbench_test_loader, writer, epoch)

        # add learning rate to tensorboard
        writer.add_scalar('LR', scheduler.get_lr()[0], epoch)

        scheduler.step()

        writer.flush()
        torch.save(model.state_dict(), os.path.join(output_path, 'models', 'epoch-'+str(epoch) + ".pt"))

        print("Epoch Time:" + str(datetime.datetime.now() - start_time))
        start_time = datetime.datetime.now()

    writer.close()


if __name__ == '__main__':
    main()
