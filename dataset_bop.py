from torch.utils.data import Dataset
import glob
import random
import os
import pickle
import json
import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image

class BOPDataset(Dataset):
    """
    Loads BOP datasets. The dataset must be parsed first using BOP_to_classification.py.
    Images needs to be of the same size. resize_BOP_dataset.py can be used to resize images.
    """
    def __init__(self, dataset_path):
        super(BOPDataset, self).__init__()
        self.dataset_path = dataset_path
        self.query_imgs_paths = os.path.join(dataset_path, 'query_good_light_only_original')
        self.gallery_imgs_paths = os.path.join(dataset_path, 'gallery_real_resized_256')

        # get number of objects in the dataset (query or gallery are the same)
        self.num_objects = len(os.listdir(self.query_imgs_paths))

        # load object images
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        image_size = [256, 256]
        self.transforms = T.Compose([
            T.Normalize(mean=pixel_mean, std=pixel_std),
            T.Resize(image_size)
        ])

    def __len__(self):
        return 1  # returns all images in the dataset

    def load_folder_images(self, folder_path, image_extension='jpg'):
        images = []
        for img_path in glob.glob(folder_path + '/*.' + image_extension):
            images.append(self.transforms(read_image(img_path).clone().float()))
        return images

    def __getitem__(self, dummy_id):
        query_imgs = []
        query_ids = []
        gallery_imgs = []
        gallery_ids = []
        # load query objects
        for query_obj_id in range(1, self.num_objects+1):
            query_imgs_paths = os.path.join(self.query_imgs_paths, f'obj_{query_obj_id:06}')
            this_obj_query_images = self.load_folder_images(query_imgs_paths, image_extension='png')
            for img in this_obj_query_images:
                query_imgs.append(img)
                query_ids.append(query_obj_id)
        # load gallery objects
        for gallery_obj_id in range(1, self.num_objects+1):
            gallery_imgs_paths = os.path.join(self.gallery_imgs_paths, f'obj_{gallery_obj_id:06}')
            this_obj_gallery_images = self.load_folder_images(gallery_imgs_paths)
            for img in this_obj_gallery_images:
                gallery_imgs.append(img)
                gallery_ids.append(gallery_obj_id)
        # stack list of tensor to tensor
        query_imgs = torch.stack(query_imgs)
        gallery_imgs = torch.stack(gallery_imgs)

        return query_imgs, query_ids, gallery_imgs, gallery_ids


def test_bop(model, device, test_loader, batch_size, epoch):
    print("Evaluating on BOP dataset")
    model.eval()

    with torch.no_grad():
        # load query and gallery images
        query_imgs, query_ids, gallery_imgs, gallery_ids = test_loader.dataset[0]

        # get embeddings - dataset is small, all images can be loaded at once
        query_embeddings = model(query_imgs)
        gallery_embeddings = model(gallery_imgs)

    # get predictions
    query_ids = np.array(query_ids)
    gallery_ids = np.array(gallery_ids)

    # get gallery ids of the same object together
    ids_unique = np.unique(gallery_ids)
    ids_unique.sort()
    ids_unique = ids_unique.tolist()
    gallery_centroids = []

    for obj_id in ids_unique:
        # get index of all images of this object
        gallery_obj_id_idx = np.where(gallery_ids == obj_id)[0]
        # get embeddings of all images of this object
        gallery_obj_id_embeddings = gallery_embeddings[gallery_obj_id_idx]
        # calculate centroid of this object
        gallery_obj_id_centroid = torch.mean(gallery_obj_id_embeddings, dim=0)
        # get embeddings of this object
        gallery_centroids.append(gallery_obj_id_centroid)

    gallery_centroids = torch.stack(gallery_centroids)

    # calculate distance matrix between query and gallery centroids
    dist_matrix = torch.cdist(query_embeddings, gallery_centroids)

    # find closest top=5 for each query object
    top_5_ids = torch.argsort(dist_matrix, dim=1)[:, :5]
    top_5_ids = top_5_ids.cpu().numpy()
    # get object ids of top 5 closest objects
    top_5_gallery_ids = []
    for i in range(len(query_ids)):
        top_5_gallery_ids.append(gallery_ids[top_5_ids[i]])
    # check if gallery_obj_id is in top_5
    rank_5 = []
    for i in range(len(query_ids)):
        if query_ids[i] in top_5_gallery_ids[i]:
            rank_5.append(1)
        else:
            rank_5.append(0)
    rank_5 = np.mean(rank_5)
    # check if gallery_obj_id is the closest
    rank_1 = []
    for i in range(len(query_ids)):
        if query_ids[i] == top_5_gallery_ids[i][0]:
            rank_1.append(1)
        else:
            rank_1.append(0)

    # calculate CMC metrics
    rank_1 = np.mean(rank_1)
    rank_5 = np.mean(rank_5)
    print("Rank 1: ", rank_1)
    print("Rank 5: ", rank_5)

