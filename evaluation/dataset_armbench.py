import glob
import random
import os
import pickle
import json
import tqdm
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class ArmBenchDataset(Dataset):
    def __init__(self, dataset_dir):
        super(ArmBenchDataset, self).__init__()
        self.query_imgs_path = os.path.join(dataset_dir, 'Picks')
        self.gallery_imgs_paths = os.path.join(dataset_dir, 'Reference_Images')
        # get images ids for train or test
        train_test_split_data = pickle.load(open(os.path.join(dataset_dir, 'train-test-split.pickle'), 'rb'))

        self.query_objs_names_list = train_test_split_data['testset']
        self.gallery_objs_names_list = train_test_split_data['testset-objects']

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
            # PickRGB is always the first image in the list
            PickRGB_path = os.path.join(self.query_imgs_path, query_obj_id, 'PickRGB.jpg')
            query_imgs_paths = glob.glob(os.path.join(self.query_imgs_path, query_obj_id) + '/*.jpg')
            # remove PickRGB from the list
            query_imgs_paths.remove(PickRGB_path)
            # join PickRGB to the beginning of the list
            query_imgs_paths.insert(0, PickRGB_path)
            self.query_imgs_paths_list.append(query_imgs_paths)

        self.gallery_objs_paths_list = {}
        # get list of images os all gallery objects
        for gallery_obj_id in self.gallery_objs_names_list:
            gallery_imgs_paths = os.path.join(self.gallery_imgs_paths, gallery_obj_id)
            self.gallery_objs_paths_list[gallery_obj_id] = glob.glob(gallery_imgs_paths + '/*.jpg')

        # object transforms
        # load object images
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        #image_size = [224, 224]
        image_size = [384, 384]
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=pixel_mean, std=pixel_std),
            T.Resize(image_size, antialias=True)
        ])

    def __len__(self):
        return len(self.query_objs_names_list)  # number of query (picks) in train or test

    def load_images(self, imgs_paths):
        """
        method will always return a list of 6 images. Just to be able to stack tensor nicely and run the whole code on GPU.
        This has no effect on the evaluation results. So the evaluation results is still correct.
        But it will end up shifting evaluation of the centroids a bit in case of objects with for 4 and 5 images.
        So it can very slightly affect the accuracy for the centroid evaluation.
        """
        imgs = [self.transforms(Image.open(img_path)) for img_path in imgs_paths]
        # make list of images always 6 images
        if len(imgs) == 1:
            imgs = imgs * 6
        elif len(imgs) == 2:
            imgs = imgs * 3
        elif len(imgs) == 3:
            imgs = imgs * 2
        elif len(imgs) == 4:
            # randomly choose 2 images and append them
            imgs = imgs + random.sample(imgs, 2)
        elif len(imgs) == 5:
            # randomly choose 1 image and append it
            imgs = imgs + random.sample(imgs, 1)
        # if 6 do nothing
        return imgs

    def __getitem__(self, query_id):
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
            # I guess there were 2 objects in the whole dataset that had no images
            # if this is not correct this would have no effect
            if len(container_obj_imgs_paths) == 0:
                continue
            else:
                container_obj_imgs = self.load_images(container_obj_imgs_paths)
                container_objs_imgs.append(container_obj_imgs)

        # Matching gallery object is always the first object in the gallery list
        return query_imgs, gallery_matching_obj_imgs, container_objs_imgs
        # dimensions: 6, 6, 6*num_of_container_objects

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


def test_armbench(device, model, test_loader, epoch):
    model.eval()

    test_loss = 0
    #criterion = CentroidTripletLoss()

    # ranking for CMC metrics - Centroid
    abs_pre_pick_rank_1, abs_pre_pick_rank_2, abs_pre_pick_rank_3 = [], [], []
    abs_pre_post_pick_rank_1, abs_pre_post_pick_rank_2, abs_pre_post_pick_rank_3 = [], [], []

    # ranking for CMC metrics - Centroid
    centroid_pre_pick_rank_1, centroid_pre_pick_rank_2, centroid_pre_pick_rank_3 = [], [], []
    centroid_pre_post_pick_rank_1, centroid_pre_post_pick_rank_2, centroid_pre_post_pick_rank_3 = [], [], []

    with torch.no_grad():
        for (all_imgs, num_gallery_objs) in tqdm.tqdm(test_loader):
            all_imgs = all_imgs.to(device)
            # run the model
            all_embeddings = model(all_imgs)
            # normalize embeddings
            # TODO check if this is needed and if query and gallery embeddings should be normalized separately?
            all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=0, p=2)
            # reshape tensor to (X,6,feat_dim) to be able to calculate centroids per object
            # view must be used instead of reshape
            feat_dim = all_embeddings.shape[-1]
            all_embeddings = all_embeddings.view((-1, 6, feat_dim))

            # iterate over each query object (batch size) to calculate CMC metrics for pre-pick with abs
            query_id = 0
            for i in range(len(num_gallery_objs)):
                # get PickRGB image embedding
                query_embedding = all_embeddings[query_id][0]
                # get gallery objects embeddings
                gallery_embeddings = all_embeddings[query_id+1:query_id+num_gallery_objs[i]+1]
                # flatten axis 0 and 1 to be able to calculate distance matrix
                gallery_embeddings = gallery_embeddings.view(-1, feat_dim)
                # increment query_id for next iteration
                query_id += num_gallery_objs[i] + 1

                # calculate distances between query and gallery objects embeddings
                dist_matrix = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), gallery_embeddings, dim=1)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)
                # map top_ids to gallery_embeddings
                top_ids = top_ids // 6

                # get rankings - find if index 0 (matching object) is in rank 1, 2 or 3
                abs_pre_pick_rank_1 += [1] if 0 in top_ids[:1] else [0]
                abs_pre_pick_rank_2 += [1] if 0 in top_ids[:2] else [0]
                abs_pre_pick_rank_3 += [1] if 0 in top_ids[:3] else [0]
            
            # iterate over each query object (batch size) to calculate CMC metrics for pre-post-pick with abs
            query_id = 0
            for i in range(len(num_gallery_objs)):  # use len(num_gallery_objs) instead of batch_size to handle last batch
                # get query object embeddings
                query_embedding = all_embeddings[query_id][0:3]
                # get gallery objects embeddings
                gallery_embeddings = all_embeddings[query_id+1:query_id+num_gallery_objs[i]+1]
                # flatten axis 0 and 1 to be able to calculate distance matrix
                gallery_embeddings = gallery_embeddings.view(-1, feat_dim)
                # increment query_id for next iteration
                query_id += num_gallery_objs[i] + 1

                # calculate distances between query and gallery objects embeddings
                dist_matrix = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(1), gallery_embeddings, dim=2)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)
                # map top_ids to gallery_embeddings
                top_ids = top_ids // 6

                # get rankings - find if index 0 (matching object) is in rank 1, 2 or 3
                abs_pre_post_pick_rank_1 += [1] if 0 in top_ids[:,0:1] else [0]
                abs_pre_post_pick_rank_2 += [1] if 0 in top_ids[:,0:2] else [0]
                abs_pre_post_pick_rank_3 += [1] if 0 in top_ids[:,0:3] else [0]

            # calculate centroids per object
            all_centroids = torch.mean(all_embeddings, dim=1)

            # iterate over each query object (batch size) to calculate CMC metrics for pre-pick with centroid
            query_id = 0
            for i in range(len(num_gallery_objs)):
                # get PickRGB image embedding
                query_embedding = all_embeddings[query_id][0]
                # get gallery objects embeddings
                gallery_centroids = all_centroids[query_id+1:query_id+num_gallery_objs[i]+1]
                # increment query_id for next iteration
                query_id += num_gallery_objs[i] + 1

                # calculate distances between query and gallery objects centroids
                dist_matrix = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), gallery_centroids, dim=1)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)

                # get rankings - find if index 0 (matching object) is in rank 1, 2 or 3
                centroid_pre_pick_rank_1 += [1] if 0 in top_ids[:1] else [0]
                centroid_pre_pick_rank_2 += [1] if 0 in top_ids[:2] else [0]
                centroid_pre_pick_rank_3 += [1] if 0 in top_ids[:3] else [0]

            # iterate over each query object (batch size) to calculate CMC metrics for pre-post-pick with centroid
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
                centroid_pre_post_pick_rank_1 += [1] if 0 in top_ids[:1] else [0]
                centroid_pre_post_pick_rank_2 += [1] if 0 in top_ids[:2] else [0]
                centroid_pre_post_pick_rank_3 += [1] if 0 in top_ids[:3] else [0]

    print("Abs =================================")
    # calculate CMC metrics for pre-pick for abs
    print("Pre-pick:")
    abs_pre_pick_rank_1, abs_pre_pick_rank_2, abs_pre_pick_rank_3 = np.mean(abs_pre_pick_rank_1), np.mean(abs_pre_pick_rank_2), np.mean(abs_pre_pick_rank_3)
    print("Rank 1: ", abs_pre_pick_rank_1)
    print("Rank 2: ", abs_pre_pick_rank_2)
    print("Rank 3: ", abs_pre_pick_rank_3)

    # calculate CMC metrics for pre-post-pick for abs
    print("Pre-post-pick:")
    abs_pre_post_pick_rank_1, abs_pre_post_pick_rank_2, abs_pre_post_pick_rank_3 = np.mean(abs_pre_post_pick_rank_1), np.mean(abs_pre_post_pick_rank_2), np.mean(abs_pre_post_pick_rank_3)
    print("Rank 1: ", abs_pre_post_pick_rank_1)
    print("Rank 2: ", abs_pre_post_pick_rank_2)
    print("Rank 3: ", abs_pre_post_pick_rank_3)

    print("Centroid =================================")
    # calculate CMC metrics for pre-pick for centroid
    print("Pre-pick:")
    centroid_pre_pick_rank_1, centroid_pre_pick_rank_2, centroid_pre_pick_rank_3 = np.mean(centroid_pre_pick_rank_1), np.mean(centroid_pre_pick_rank_2), np.mean(centroid_pre_pick_rank_3)
    print("Rank 1: ", centroid_pre_pick_rank_1)
    print("Rank 2: ", centroid_pre_pick_rank_2)
    print("Rank 3: ", centroid_pre_pick_rank_3)

    # calculate CMC metrics for pre-post-pick for centroid
    print("Pre-post-pick:")
    centroid_pre_post_pick_rank_1, centroid_pre_post_pick_rank_2, centroid_pre_post_pick_rank_3 = np.mean(centroid_pre_post_pick_rank_1), np.mean(centroid_pre_post_pick_rank_2), np.mean(centroid_pre_post_pick_rank_3)
    print("Rank 1: ", centroid_pre_post_pick_rank_1)
    print("Rank 2: ", centroid_pre_post_pick_rank_2)
    print("Rank 3: ", centroid_pre_post_pick_rank_3)
