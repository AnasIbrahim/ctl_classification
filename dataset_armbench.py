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
        image_size = [224, 224]
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
        annotation_json = json.load(
            open(os.path.join(self.query_imgs_paths, self.query_objects_ids[query_id], 'annotation.json'), 'r'))
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
            container_json = json.load(
                open(os.path.join(self.query_imgs_paths, self.query_objects_ids[query_id], 'container.json'), 'r'))
            container_objs_ids = container_json.keys()
            # load all images of all objects in pick_id
            gallery_objs_imgs = [gallery_imgs]  # first object in the gallery is the matching query object
            gallery_objs_names = [gallery_matching_id]
            gallery_objs_local_ids = [0] * len(gallery_imgs)
            obj_local_id = 1  # counting starts from 1, 0 is the matching matching gallery object
            for obj_name in container_objs_ids:
                if obj_name == gallery_matching_id:
                    continue
                gallery_imgs_paths = os.path.join(self.gallery_imgs_paths, obj_name)
                this_obj_imgs = self.load_folder_images(gallery_imgs_paths)
                # check if this_obj_imgs is empty
                # TODO check if this is the same for the evaluation in the ArmBench paper
                if len(this_obj_imgs) == 0:
                    continue
                else:
                    gallery_objs_imgs.append(this_obj_imgs)
                    gallery_objs_names.append(obj_name)
                    gallery_objs_local_ids += [obj_local_id] * len(this_obj_imgs)
                    obj_local_id += 1

            gallery_objs_imgs = torch.cat(gallery_objs_imgs)
            gallery_objs_local_ids = torch.tensor(gallery_objs_local_ids)

            # Matching gallery object is always the first object in the gallery list
            return query_imgs, gallery_objs_imgs, gallery_objs_local_ids, gallery_objs_names, gallery_matching_id

    @staticmethod
    def collate_fn(batch):
        return  batch

def test_armbench(model, device, test_loader, batch_size, epoch):
    print("Evaluating model")
    model.eval()

    test_loss = 0
    #criterion = CentroidTripletLoss()

    # ranking for CMC metrics
    rank_1, rank_2, rank_3 = [], [], []

    with torch.no_grad():
        for batches_data in tqdm.tqdm(test_loader):
            # combine query and gallery images from all samples in the batch and presrve the ids
            batch_query_imgs, batch_gallery_objs_imgs = [], []
            batch_query_ids, batch_gallery_ids = [], []
            for batch_id, batch_data in enumerate(batches_data):
                query_imgs, gallery_objs_imgs, gallery_objs_local_ids, gallery_objs_names, gallery_matching_id = batch_data
                query_imgs, gallery_objs_imgs = query_imgs.to(device), gallery_objs_imgs.to(device)
                batch_query_imgs += [query_imgs]
                batch_gallery_objs_imgs += [gallery_objs_imgs]
                batch_query_ids += [batch_id] * len(query_imgs)
                batch_gallery_ids += [batch_id] * len(gallery_objs_imgs)

            batch_query_imgs = torch.cat(batch_query_imgs)
            batch_gallery_objs_imgs = torch.cat(batch_gallery_objs_imgs)
            batch_query_ids = torch.tensor(batch_query_ids)
            batch_gallery_ids = torch.tensor(batch_gallery_ids)

            # combine query and gallery together
            all_imgs = torch.cat([batch_query_imgs, batch_gallery_objs_imgs])
            # runt the model
            all_embeddings = model(all_imgs)

            # separate query and gallery embeddings
            batch_query_embeddings = all_embeddings[:len(batch_query_imgs)]
            batch_gallery_obj_embeddings = all_embeddings[len(batch_query_imgs):]

            # normalize embeddings
            # TODO should we normalize the query and gallery embeddings together?
            batch_query_embeddings = torch.nn.functional.normalize(batch_query_embeddings, dim=1, p=2)
            batch_gallery_obj_embeddings = torch.nn.functional.normalize(batch_gallery_obj_embeddings, dim=1, p=2)

            for batch_id, batch_data in enumerate(batches_data):
                _, _, gallery_objs_local_ids, gallery_objs_names, gallery_matching_id = batch_data
                # separate each batch embeddings
                query_embeddings = batch_query_embeddings[batch_query_ids == batch_id]
                gallery_obj_embeddings = batch_gallery_obj_embeddings[batch_gallery_ids == batch_id]

                # calculate centroids
                query_centroid = torch.mean(query_embeddings, dim=0)
                # calculate centroids for each gallery object
                gallery_objs_centroids = []
                for i in range(len(gallery_objs_names)):
                    this_obj_gallery_embeddings = gallery_obj_embeddings[gallery_objs_local_ids == i]
                    this_obj_gallery_centroid = torch.mean(this_obj_gallery_embeddings, dim=0)
                    gallery_objs_centroids.append(this_obj_gallery_centroid)
                gallery_objs_centroids = torch.stack(gallery_objs_centroids)

                # calculate distances between gallery centroid and query objects centroids
                dist_matrix = torch.nn.functional.cosine_similarity(query_centroid.unsqueeze(0), gallery_objs_centroids, dim=1)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)
                # get object ids of the closest objects
                top_ids = [gallery_objs_names[i] for i in top_ids]
                # get rankings
                rank_1 += [1] if gallery_matching_id in top_ids[:1] else [0]
                rank_2 += [1] if gallery_matching_id in top_ids[:2] else [0]
                rank_3 += [1] if gallery_matching_id in top_ids[:3] else [0]

    # calculate CMC metrics
    rank_1, rank_2, rank_3 = np.mean(rank_1), np.mean(rank_2), np.mean(rank_3)
    print("Rank 1: ", rank_1)
    print("Rank 2: ", rank_2)
    print("Rank 3: ", rank_3)
