import argparse, random, copy
import os
import datetime
import numpy as np
import pickle
import cv2
import glob
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR

from torchvision.io import read_image
from torchvision.utils import save_image

#class CTLModel(nn.Module):
#    def __init__(self):
#        super(CTLModel, self).__init__()
#        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
#
#    def forward(self, img):
#        output = self.resnet(img)
#        output = output.view(output.size()[0], -1)
#        return output


class ArmBenchDataset(Dataset):
    def __init__(self, training_dir, mode='train'):
        super(ArmBenchDataset, self).__init__()
        self.training_dir = training_dir
        self.query_imgs_paths = os.path.join(training_dir, 'Picks')
        self.gallery_imgs_paths = os.path.join(training_dir, 'Reference_Images')
        self.mode = mode
        # get images ids for train or test
        train_test_split_data = pickle.load(open(os.path.join(training_dir, 'train-test-split.pickle'), 'rb'))
        if self.mode == 'train':
            self.query_objects_ids = train_test_split_data['trainset']
            self.gallery_objects_list = train_test_split_data['trainset-objects']
        elif self.mode == 'test':
            self.query_objects_ids = train_test_split_data['testset']
            self.gallery_objects_list = train_test_split_data['testset-objects']

        # load object images
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        image_size = [256, 256]
        self.transforms = T.Compose([
            T.Normalize(mean=pixel_mean, std=pixel_std),
            T.Resize(image_size)
        ])

    def __len__(self):
        return len(self.query_objects_ids)  # number of query (picks) in train or test

    def load_folder_images(self, folder_path):
        images = []
        for img_path in glob.glob(folder_path + '/*.jpg'):
            images.append(self.transforms(read_image(img_path).clone().float()))
        if len(images) == 0:
            return []
        # convert images list to tensor
        images = torch.stack(images)
        return images
    
    def load_sample_images(self, query_id, query=False):
        # read annotation.json
        annotation_json = json.load(open(os.path.join(self.query_imgs_paths, self.query_objects_ids[query_id], 'annotation.json'), 'r'))
        # get object id of pick_id from annotation.json
        gallery_obj_id = annotation_json['GT_ID']
        gallery_imgs_paths = os.path.join(self.gallery_imgs_paths, gallery_obj_id)
        gallery_imgs = self.load_folder_images(gallery_imgs_paths)

        if not query:
            return gallery_imgs, gallery_obj_id
        if query:
            query_imgs_paths = os.path.join(self.query_imgs_paths, self.query_objects_ids[query_id])
            query_imgs = self.load_folder_images(query_imgs_paths)

            return query_imgs, gallery_imgs, gallery_obj_id

    def __getitem__(self, query_id):
        if self.mode == 'train':
            anchor_imgs, positive_imgs, positive_obj_id = self.load_sample_images(query_id, query=True)
            # choose a random id from obj_ids as negative, make sure negative_obj_id is not the same as positive_obj_id
            while True:
                negative_obj_id = random.randint(0, len(self.gallery_objects_list) - 1)
                negative_obj_id = self.gallery_objects_list[negative_obj_id]
                if negative_obj_id != positive_obj_id:
                    break
            negative_imgs, _ = self.load_sample_images(negative_obj_id, query=False)
            return anchor_imgs, positive_imgs, negative_imgs
        elif self.mode == 'test':
            # load query images
            query_imgs, gallery_imgs, gallery_matching_id = self.load_sample_images(query_id, query=True)
            # get object id of all object in pick_id from container.json
            container_json = json.load(open(os.path.join(self.query_imgs_paths, self.query_objects_ids[query_id], 'container.json'), 'r'))
            container_objs_ids = container_json.keys()
            # load all images of all objects in pick_id
            gallery_objs_imgs = [gallery_imgs]  # first object in the gallery is the matching query object
            gallery_objs_ids = [gallery_matching_id]
            for obj_id in container_objs_ids:
                if obj_id == gallery_matching_id:
                    continue
                gallery_imgs_paths = os.path.join(self.gallery_imgs_paths, obj_id)
                this_obj_imgs = self.load_folder_images(gallery_imgs_paths)
                # check if this_obj_imgs is empty
                # TODO check if this is the same for the evaluation in the ArmBench paper
                if len(this_obj_imgs) == 0:
                    continue
                gallery_objs_imgs.append(this_obj_imgs)
                gallery_objs_ids.append(obj_id)
            return query_imgs, gallery_objs_imgs, gallery_objs_ids, gallery_matching_id


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


def test_armbench(model, device, test_loader, epoch):
    print("Evaluating model")
    model.eval()

    test_loss = 0
    #criterion = CentroidTripletLoss()

    # ranking for CMC metrics
    rank_1 = []
    rank_5 = []

    with torch.no_grad():
        for query_id in range(len(test_loader.dataset)):
            # get query and gallery images
            query_imgs, gallery_objs_imgs, gallery_objs_ids, gallery_matching_id = test_loader.dataset[query_id]
            query_imgs, gallery_objs_imgs = query_imgs.to(device), [gallery_imgs.to(device) for gallery_imgs in gallery_objs_imgs]
            # get embeddings
            query_embeddings = model(query_imgs)
            gallery_obj_embeddings = [model(gallery_imgs) for gallery_imgs in gallery_objs_imgs]

            # calculate centroids
            query_centroid = torch.mean(query_embeddings, dim=0)
            gallery_objs_centroids = [torch.mean(gallery_embeddings, dim=0) for gallery_embeddings in gallery_obj_embeddings]

            # calculate distances between gallery centroid and query objects centroids
            distances = [torch.dist(query_centroid, gallery_centroid).cpu().numpy() for gallery_centroid in gallery_objs_centroids]

            # get top 5 closest objects
            top_5 = np.argsort(distances)[:5]  # TODO handle cases where there is less than 5 objects in gallery
            # get object ids of top 5 closest objects
            top_5 = [gallery_objs_ids[i] for i in top_5]
            # check if gallery_obj_id is in top_5
            if gallery_matching_id in top_5:
                rank_5.append(1)
            else:
                rank_5.append(0)
            # check if gallery_obj_id is the closest
            if gallery_matching_id == top_5[0]:
                rank_1.append(1)
            else:
                rank_1.append(0)

    # calculate CMC metrics
    rank_1 = np.mean(rank_1)
    rank_5 = np.mean(rank_5)
    print("Rank 1: ", rank_1)
    print("Rank 5: ", rank_5)


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
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
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
