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

class ArmBenchDataset(Dataset):
    def __init__(self, training_dir, mode='train', portion=None):
        super(ArmBenchDataset, self).__init__()
        self.training_dir = training_dir
        self.query_imgs_path = os.path.join(training_dir, 'Picks')
        self.gallery_imgs_paths = os.path.join(training_dir, 'Reference_Images')
        self.mode = mode
        # get images ids for train or test
        train_test_split_data = pickle.load(open(os.path.join(training_dir, 'train-test-split.pickle'), 'rb'))
        if self.mode == 'train':
            self.query_objs_names_list = train_test_split_data['trainset']
            self.gallery_objs_names_list = train_test_split_data['trainset-objects']
        elif self.mode == 'test':
            self.query_objs_names_list = train_test_split_data['testset']
            self.gallery_objs_names_list = train_test_split_data['testset-objects']

        # if portion is set, use only a small portion of the dataset
        if portion:
            self.query_objs_names_list = self.query_objs_names_list[:portion]

        # for each query object get associated gallery object and list of object in container
        self.query_imgs_paths_list = []
        self.query_associated_gallery_object = []
        self.query_associated_container_objects = []
        for query_obj_id in self.query_objs_names_list:
            # read annotation.json
            annotation_json = json.load(open(os.path.join(self.query_imgs_path, query_obj_id, 'annotation.json'), 'r'))
            # get object id of pick_id from annotation.json
            gallery_obj_id = annotation_json['GT_ID']
            self.query_associated_gallery_object.append(gallery_obj_id)

            # get object id of all object in pick_id from container.json
            container_json = json.load(open(os.path.join(self.query_imgs_path, query_obj_id, 'container.json'), 'r'))
            self.query_associated_container_objects.append(list(container_json.keys()))

            # list images in query the object's folder
            self.query_imgs_paths_list.append(glob.glob(os.path.join(self.query_imgs_path, query_obj_id) + '/*.jpg'))

        # if portions set - use objects associated with query as gallery. This is to test training only
        #if portion:
        #    self.gallery_objs_names_list = self.query_associated_gallery_object  # can only test training with this line

        self.gallery_objs_paths_list = {}
        # get list of images os all gallery objects
        for gallery_obj_id in self.gallery_objs_names_list:
            gallery_imgs_paths = os.path.join(self.gallery_imgs_paths, gallery_obj_id)
            self.gallery_objs_paths_list[gallery_obj_id] = glob.glob(gallery_imgs_paths + '/*.jpg')

        # object transforms
        # load object images
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        image_size = [224, 224]
        self.transforms = T.Compose([
            T.Normalize(mean=pixel_mean, std=pixel_std),
            T.Resize(image_size)
        ])

    def __len__(self):
        return len(self.query_objs_names_list)  # number of query (picks) in train or test

    def load_images(self, imgs_paths):
        """
        method will always return a list of 6 images. Just to be able to stack tensor nicely and run the whole code on GPU
        """
        imgs = [self.transforms(read_image(img_path).clone().float()) for img_path in imgs_paths]
        # make list of images always 6 images
        if len(imgs) == 1:
            imgs = imgs * 6
        elif len(imgs) == 2:
            imgs = imgs * 3
        elif len(imgs) == 3:
            imgs = imgs * 2
        # NOTE: this will end up shifting the centroid of this object a bit for 4 and 5 images
        # Still this is crucial for the training to be done efficiently
        # TODO try training without 4 and 5 and check if this gets a better result
        elif len(imgs) == 4:
            # randomly choose 2 images and append them
            imgs = imgs + random.sample(imgs, 2)
        elif len(imgs) == 5:
            # randomly choose 1 image and append it
            imgs = imgs + random.sample(imgs, 1)
        # if 6 do nothing

        return imgs

    def __getitem__(self, query_id):
        if self.mode == 'train':
            anchor_imgs = self.load_images(self.query_imgs_paths_list[query_id])
            positive_obj_id = self.query_associated_gallery_object[query_id]
            positive_imgs = self.load_images(self.gallery_objs_paths_list[positive_obj_id])
            # choose a random id from obj_ids as negative, make sure negative_obj_id is not the same as positive_obj_id
            while True:
                negative_obj_id = random.randint(0, len(self.gallery_objs_names_list) - 1)
                negative_obj_id = self.gallery_objs_names_list[negative_obj_id]
                # check if negative_obj_id is not empty
                if len(self.gallery_objs_paths_list[negative_obj_id]) == 0:
                    continue
                # make sure it is not the same object as positive_obj_id
                if negative_obj_id != positive_obj_id:
                    break
            negative_imgs = self.load_images(self.gallery_objs_paths_list[negative_obj_id])
            return anchor_imgs, positive_imgs, negative_imgs
        elif self.mode == 'test':
            # load query images
            query_imgs = self.load_images(self.query_imgs_paths_list[query_id])
            gallery_matching_obj_name = self.query_associated_gallery_object[query_id]
            gallery_matching_obj_imgs = self.load_images(self.gallery_objs_paths_list[gallery_matching_obj_name])
            # load all images of all objects in pick_id
            container_objs_imgs = []
            for container_obj_name in self.query_associated_container_objects[query_id]:
                if container_obj_name == gallery_matching_obj_name:
                    continue
                container_obj_imgs_paths = self.gallery_objs_paths_list[container_obj_name]
                # if container object has no images, skip it
                # TODO check if this is the same for the evaluation in the ArmBench paper
                if len(container_obj_imgs_paths) == 0:
                    continue
                else:
                    container_obj_imgs = self.load_images(container_obj_imgs_paths)
                    container_objs_imgs.append(container_obj_imgs)

            # Matching gallery object is always the first object in the gallery list
            return query_imgs, gallery_matching_obj_imgs, container_objs_imgs
            # dimensions: 6, 6, 6*num_of_container_objects

    @staticmethod
    def train_collate_fn(batch):
        all_imgs = []
        for anchor_imgs, positive_imgs, negative_imgs in batch:
            for imgs in [anchor_imgs, positive_imgs, negative_imgs]:
                all_imgs.extend(imgs)
        # convert list to tensor
        all_imgs = torch.stack(all_imgs)
        return all_imgs

    @staticmethod
    def test_collate_fn(batch):
        all_imgs = []
        num_gallery_objs = []
        for query_imgs, gallery_matching_obj_imgs, container_objs_imgs in batch:
            all_imgs.extend(query_imgs)
            all_imgs.extend(gallery_matching_obj_imgs)
            for container_obj_imgs in container_objs_imgs:
                all_imgs.extend(container_obj_imgs)
            num_gallery_objs.append(len(container_objs_imgs)+1)  # matching gallery object + other objects in the container
        # convert list to tensor
        all_imgs = torch.stack(all_imgs)
        return all_imgs, num_gallery_objs


def test_armbench(model, test_loader):
    model.eval()

    test_loss = 0
    #criterion = CentroidTripletLoss()

    # ranking for CMC metrics
    rank_1, rank_2, rank_3 = [], [], []

    with torch.no_grad():
        for (all_imgs, num_gallery_objs) in tqdm.tqdm(test_loader):
            # run the model
            all_embeddings = model(all_imgs)
            # normalize embeddings
            # TODO check if this is needed and if query and gallery embeddings should be normalized separately?
            all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=0, p=2)
            # reshape tensor to (X,6,feat_dim) to be able to calculate centroids per object
            # view must be used instead of reshape
            feat_dim = all_embeddings.shape[-1]
            all_embeddings = all_embeddings.view((-1, 6, feat_dim))
            # calculate centroids per object
            all_centroids = torch.mean(all_embeddings, dim=1)

            # iterate over each query object (batch size)
            query_id = 0
            for i in range(len(num_gallery_objs)):  # use len(num_gallery_objs) instead of batch_size to handle last batch
                # get query object embeddings
                query_centroid = all_centroids[query_id]
                # get gallery objects embeddings
                gallery_centroids = all_centroids[query_id+1:query_id+num_gallery_objs[i]+1]
                # increment query_id for next iteration
                query_id += num_gallery_objs[i] + 1

                # calculate distances between query and gallery objects centroids
                dist_matrix = torch.nn.functional.cosine_similarity(query_centroid.unsqueeze(0), gallery_centroids, dim=1)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)

                # get rankings - find if index 0 (matching object) is in rank 1, 2 or 3
                rank_1 += [1] if 0 in top_ids[:1] else [0]
                rank_2 += [1] if 0 in top_ids[:2] else [0]
                rank_3 += [1] if 0 in top_ids[:3] else [0]

    # calculate CMC metrics
    rank_1, rank_2, rank_3 = np.mean(rank_1), np.mean(rank_2), np.mean(rank_3)
    print("Rank 1: ", rank_1)
    print("Rank 2: ", rank_2)
    print("Rank 3: ", rank_3)
