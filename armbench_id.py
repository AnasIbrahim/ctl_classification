
"""
The ARMBench object identification dataset
"""

import torch
import numpy as np
import torchvision.transforms

from pathlib import Path
from dataclasses import dataclass
import json
import random
import tempfile
import pickle
from tqdm import tqdm
from PIL import Image
import fcntl

@dataclass
class Pick:
    name: str
    object: str
    other_objects: list
    pick_images: int
    object_images: int
    total_images: int

class ARMBench:
    def __init__(self, path, split, batch_size=512, augment=False):
        super().__init__()

        assert split in ['train', 'test']
        #assert situation in ['pre-pick', 'post-pick']

        self.path = Path(path)
        self.batch_size = batch_size
        self.split = split
        # self.situation = situation

        self.reference_dir = self.path / "Reference_Images"
        self.pick_dir = self.path / "Picks"

        # Transforms
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        # self.image_size = [224, 224]
        self.image_size = [384, 384]
        self.transforms = []
        if augment:
            self.transforms.append(torchvision.transforms.TrivialAugmentWide())
        self.transforms += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=pixel_mean, std=pixel_std),
            torchvision.transforms.Resize(self.image_size, antialias=True)
        ]

        self.transforms = torchvision.transforms.Compose(self.transforms)

        with open(self.path / f"lock", 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)

            # If we have a cache file, load it
            cache_file = self.path / f"cache-{split}-v2.pth"
            if cache_file.exists():
                cache = torch.load(cache_file)
                self.objects = cache['objects']
                self.picks = cache['picks']
            else:
                split_file = self.path / "train-test-split.pickle"
                with open(split_file, 'rb') as f:
                    split_data = pickle.load(f)

                self._build_lists(split_picks=split_data[f'{split}set'], split_objects=split_data[f'{split}set-objects'])

                tmpfile = tempfile.NamedTemporaryFile(dir=self.path, delete=False)
                torch.save({
                    'objects': self.objects,
                    'picks': self.picks,
                    'version': 1,
                }, tmpfile.name)
                Path(tmpfile.name).rename(cache_file)

        self.object_list = list(self.objects.items())

        self.rng = random.Random(0)

    def epoch_view(self):
        return ARMBenchEpochView(self, self.batch_size)

    def test_view(self, *args, **kwargs):
        return ARMBenchTestView(self, *args, **kwargs)

    def _build_lists(self, split_picks, split_objects):
        # List all reference images
        self.objects = {}
        for obj in split_objects:
            obj_path = self.reference_dir / obj
            if not obj_path.is_dir():
                continue

            num_images = len([ p for p in obj_path.iterdir() if p.suffix == '.jpg'])

            if num_images == 0:
                continue

            self.objects[obj_path.name] = num_images

        # List all picks
        self.picks = []
        for pick in split_picks:
            pick_path = self.pick_dir / pick
            json_path = pick_path / "annotation.json"
            if not json_path.exists():
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            object = data['GT_ID']
            num_images = len([ p for p in pick_path.iterdir() if p.suffix == '.jpg'])

            if num_images == 0:
                continue

            object_images = self.objects.get(object)
            if object_images is None:
                continue

            container_path = pick_path / "container.json"
            if not container_path.exists():
                continue

            with open(container_path, 'r') as f:
                container_data = json.load(f)

            other_objects = [ o for o in container_data.keys() if object != o and o in self.objects ]

            self.picks.append(Pick(
                name=pick_path.name,
                object=object,
                other_objects=other_objects,
                pick_images=num_images,
                object_images=object_images,
                total_images=num_images + object_images
            ))

        print(f"Loaded ARMBench dataset with {len(self.objects)} reference objects and {len(self.picks)} picks")

class ARMBenchEpochView(torch.utils.data.Dataset):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        batch_size = dataset.batch_size

        # Build batches!

        # Retrieve a shuffled copy of the picks
        refs = dataset.picks.copy()

        if shuffle:
            self.dataset.rng.shuffle(refs)

        batch = []
        cur_batch_size = 0

        batches = []

        obj_idx = 0

        rejects = []

        # At maximum we have 4 pick images, 6 positive object images, and 6 negative object images
        MAX_IMAGES_PER_PICK = 4 + 6 + 6

        while obj_idx != len(refs):
            remaining = batch_size - cur_batch_size

            # If there is enough space for at least one, add picks!
            if remaining >= MAX_IMAGES_PER_PICK:
                min_items = min(remaining // MAX_IMAGES_PER_PICK, len(refs) - obj_idx)

                for i in range(min_items):
                    pick = refs[obj_idx + i]

                    # Choose a negative
                    neg_obj_name, neg_obj_size = random.choice(self.dataset.object_list)

                    batch.append((pick, neg_obj_name))
                    cur_batch_size += pick.total_images + neg_obj_size

                obj_idx += min_items
            else:
                batches.append(batch)
                batch = []
                cur_batch_size = 0

        if len(batch) != 0:
            batches.append(batch)

        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]

        images = torch.zeros(self.dataset.batch_size, 3, *self.dataset.image_size)
        image_idx = 0

        # For each image in images, this contains the aggregation index
        agg_indices = torch.zeros(self.dataset.batch_size, dtype=torch.long)

        # For every aggregation index, this is the number of images aggregated in it
        agg_counts = torch.ones(1 + 3*self.dataset.batch_size, dtype=torch.long)

        # For every object, this is the first image (the PickRGB image)
        image_index = torch.zeros(len(batch), dtype=torch.long)

        # Ref 0 is reserved for the unfilled image slots
        agg_idx = 1

        for i, batch_entry in enumerate(batch):
            pick, neg_obj_name = batch_entry
            image_index[i] = image_idx

            # Load pick images
            pick_path = self.dataset.pick_dir / pick.name
            pick_images = [pick_path / "PickRGB.jpg"]
            pick_images += sorted([ p for p in pick_path.iterdir() if p.suffix == '.jpg' and p.name != "PickRGB.jpg" ])
            assert len(pick_images) == pick.pick_images, f"Wrong pick_images for {pick.name}: cached {pick.pick_images}, found: {pick_images}"

            for path in pick_images:
                images[image_idx] = self.dataset.transforms(Image.open(path))
                agg_indices[image_idx] = agg_idx
                agg_counts[agg_idx] += 1
                image_idx += 1

            agg_idx += 1

            # Load positive object images
            pos_path = self.dataset.reference_dir / pick.object
            for i in range(pick.object_images):
                path = pos_path / f"{i+1}.jpg"
                images[image_idx] = self.dataset.transforms(Image.open(path))
                agg_indices[image_idx] = agg_idx
                agg_counts[agg_idx] += 1
                image_idx += 1

            agg_idx += 1

            # Load negative object images
            neg_path = self.dataset.reference_dir / neg_obj_name
            for i in range(self.dataset.objects[neg_obj_name]):
                path = neg_path / f"{i+1}.jpg"
                images[image_idx] = self.dataset.transforms(Image.open(path))
                agg_indices[image_idx] = agg_idx
                agg_counts[agg_idx] += 1
                image_idx += 1

            agg_idx += 1

        agg_counts = agg_counts[:agg_idx].clone()

        return {
            'images': images,
            'agg_indices': agg_indices,
            'agg_counts': agg_counts,
            'image_index': image_index,
        }

class ARMBenchTestView(torch.utils.data.Dataset):
    def __init__(self, dataset, subsample=1):
        self.dataset = dataset
        batch_size = self.dataset.batch_size

        assert self.dataset.split == 'test'

        # Build batches
        # Retrieve a shuffled copy of the picks
        batch = []
        cur_batch_size = 0

        batches = []

        obj_idx = 0

        for pick in self.dataset.picks[::subsample]:
            size = pick.total_images
            size += sum([ self.dataset.objects[o] for o in pick.other_objects ])

            if cur_batch_size + size > batch_size:
                batches.append(batch)
                batch = []
                cur_batch_size = 0

            batch.append(pick)
            cur_batch_size += size
            assert cur_batch_size <= batch_size

        if len(batch) != 0:
            batches.append(batch)

        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]

        images = torch.zeros(self.dataset.batch_size, 3, *self.dataset.image_size)
        image_idx = 0

        # For each image in images, this contains the aggregation index
        agg_indices = torch.zeros(self.dataset.batch_size, dtype=torch.long)

        # For every aggregation index, this is the number of images aggregated in it
        agg_counts = torch.ones(1 + (1+10)*self.dataset.batch_size, dtype=torch.long)

        # For every object, this is the number of "other" objects
        other_obj_counts = torch.zeros(len(batch), dtype=torch.long)

        # For every object, this is the "start" index in images
        image_index = torch.zeros(len(batch), dtype=torch.long)

        # Ref 0 is reserved for the unfilled image slots
        agg_idx = 1

        for pick_idx, pick in enumerate(batch):
            image_index[pick_idx] = image_idx
            other_obj_counts[pick_idx] = len(pick.other_objects)

            # Load pick images
            pick_path = self.dataset.pick_dir / pick.name
            pick_images = [pick_path / "PickRGB.jpg"]
            pick_images += sorted([ p for p in pick_path.iterdir() if p.suffix == '.jpg' and p.name != "PickRGB.jpg" ])
            assert len(pick_images) == pick.pick_images, f"Wrong pick_images for {pick.name}: cached {pick.pick_images}, found: {pick_images}"

            for path in pick_images:
                images[image_idx] = self.dataset.transforms(Image.open(path))
                agg_indices[image_idx] = agg_idx
                agg_counts[agg_idx] += 1
                image_idx += 1

            agg_idx += 1

            # Load positive object images
            pos_path = self.dataset.reference_dir / pick.object
            for i in range(pick.object_images):
                path = pos_path / f"{i+1}.jpg"
                images[image_idx] = self.dataset.transforms(Image.open(path))
                agg_indices[image_idx] = agg_idx
                agg_counts[agg_idx] += 1
                image_idx += 1

            agg_idx += 1

            # Load other object images
            for obj in pick.other_objects:
                neg_path = self.dataset.reference_dir / obj
                for i in range(self.dataset.objects[obj]):
                    path = neg_path / f"{i+1}.jpg"
                    images[image_idx] = self.dataset.transforms(Image.open(path))
                    agg_indices[image_idx] = agg_idx
                    agg_counts[agg_idx] += 1
                    image_idx += 1

                agg_idx += 1

        agg_counts = agg_counts[:agg_idx].clone()

        return {
            'images': images,
            'agg_indices': agg_indices,
            'agg_counts': agg_counts,
            'other_obj_counts': other_obj_counts,
            'image_index': image_index
        }

def evaluate(model, test_loader, is_distributed=False, disable_progress=False):
    test_loss = 0

    # ranking for CMC metrics
    sample_count = 0
    pre_pick_rank_1 = 0
    pre_pick_rank_2 = 0
    pre_pick_rank_3 = 0
    post_pick_rank_1 = 0
    post_pick_rank_2 = 0
    post_pick_rank_3 = 0
    instance_pre_pick_rank_1 = 0
    instance_pre_pick_rank_2 = 0
    instance_pre_pick_rank_3 = 0
    instance_post_pick_rank_1 = 0
    instance_post_pick_rank_2 = 0
    instance_post_pick_rank_3 = 0

    with torch.inference_mode():
        for batch in tqdm(test_loader, disable=disable_progress):
            images = batch['images'].cuda()
            agg_indices = batch['agg_indices'].cuda()
            agg_counts = batch['agg_counts'].cuda()
            other_object_counts = batch['other_obj_counts']
            image_index = batch['image_index']
            num_picks = other_object_counts.shape[0]

            # run the model
            all_embeddings = model(images)

            # TODO: required?
            #all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=-1, p=2)

            # Add up all image embeddings in their aggregation slots
            aggregated = torch.zeros(agg_counts.shape[0], all_embeddings.shape[1], device=all_embeddings.device)
            aggregated.index_add_(dim=0, index=agg_indices, source=all_embeddings)

            # Divide by the count to compute the mean
            aggregated /= agg_counts.unsqueeze(1)

            # Ignore aggregation slot 0
            aggregated = aggregated[1:]
            agg_counts = agg_counts[1:]

            # pre-pick
            aggregated_offset = 0
            for i in range(num_picks):
                # get PickRGB image embedding
                query_embedding = all_embeddings[image_index[i]]

                # get gallery object embeddings
                gallery_centroids = aggregated[aggregated_offset+1:aggregated_offset+1+1+other_object_counts[i]]

                # calculate distances between query and gallery objects centroids
                dist_matrix = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), gallery_centroids, dim=1)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)

                # get rankings - find if index 0 (matching object) is in rank 1, 2 or 3
                pre_pick_rank_1 += 1 if 0 in top_ids[:1] else 0
                pre_pick_rank_2 += 1 if 0 in top_ids[:2] else 0
                pre_pick_rank_3 += 1 if 0 in top_ids[:3] else 0

                aggregated_offset += 1 + 1 + other_object_counts[i]
                sample_count += 1

            # instance pre-pick
            aggregated_offset = 0
            for i in range(num_picks):
                # get PickRGB image embedding
                query_embedding = all_embeddings[image_index[i]]

                # get gallery object embeddings
                num_query_imgs = agg_counts[aggregated_offset + 0].item()
                num_positive_imgs = agg_counts[aggregated_offset + 1].item()
                num_other_imgs = agg_counts[aggregated_offset + 2:aggregated_offset + 2 + other_object_counts[i]].sum().item()

                gallery_instances = all_embeddings[image_index[i] + num_query_imgs:image_index[i] + num_query_imgs + num_positive_imgs + num_other_imgs]

                # calculate distances between query and gallery objects centroids
                dist_matrix = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), gallery_instances, dim=1)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)

                # get rankings - find if index 0 (matching object) is in rank 1, 2 or 3
                instance_pre_pick_rank_1 += 1 if top_ids[0] < num_positive_imgs else 0
                instance_pre_pick_rank_2 += 1 if np.min(top_ids[:2]) < num_positive_imgs else 0
                instance_pre_pick_rank_3 += 1 if np.min(top_ids[:3]) < num_positive_imgs else 0

                aggregated_offset += 1 + 1 + other_object_counts[i]

            # post-pick
            aggregated_offset = 0
            for i in range(num_picks):
                # get query object embeddings
                query_centroid = aggregated[aggregated_offset+0]
                # get gallery objects embeddings
                gallery_centroids = aggregated[aggregated_offset+1:aggregated_offset+1+1+other_object_counts[i]]

                # calculate distances between query and gallery objects centroids
                dist_matrix = torch.nn.functional.cosine_similarity(query_centroid.unsqueeze(0), gallery_centroids, dim=1)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)

                # get rankings - find if index 0 (matching object) is in rank 1, 2 or 3
                post_pick_rank_1 += 1 if 0 in top_ids[:1] else 0
                post_pick_rank_2 += 1 if 0 in top_ids[:2] else 0
                post_pick_rank_3 += 1 if 0 in top_ids[:3] else 0

                aggregated_offset += 1 + 1 + other_object_counts[i]

            # instance post-pick
            aggregated_offset = 0
            for i in range(num_picks):
                # get query object embeddings
                query_centroid = aggregated[aggregated_offset+0]

                # get gallery object embeddings
                num_query_imgs = agg_counts[aggregated_offset + 0].item()
                num_positive_imgs = agg_counts[aggregated_offset + 1].item()
                num_other_imgs = agg_counts[aggregated_offset + 2:aggregated_offset + 2 + other_object_counts[i]].sum().item()

                gallery_instances = all_embeddings[image_index[i] + num_query_imgs:image_index[i] + num_query_imgs + num_positive_imgs + num_other_imgs]

                # calculate distances between query and gallery objects centroids
                dist_matrix = torch.nn.functional.cosine_similarity(query_centroid.unsqueeze(0), gallery_instances, dim=1)
                dist_matrix = 1 - dist_matrix
                dist_matrix = dist_matrix.cpu().numpy()

                # sort closest objects
                top_ids = np.argsort(dist_matrix)

                # get rankings - find if index 0 (matching object) is in rank 1, 2 or 3
                instance_post_pick_rank_1 += 1 if top_ids[0] < num_positive_imgs else 0
                instance_post_pick_rank_2 += 1 if np.min(top_ids[:2]) < num_positive_imgs else 0
                instance_post_pick_rank_3 += 1 if np.min(top_ids[:3]) < num_positive_imgs else 0

                aggregated_offset += 1 + 1 + other_object_counts[i]

    if is_distributed:
        def gather_sum(x):
            if torch.distributed.get_rank() == 0:
                out = [ None for i in range(torch.distributed.get_world_size()) ]
                torch.distributed.gather_object(x, object_gather_list=out)
                return sum(out)
            else:
                torch.distributed.gather_object(x)

        # Bring everything to rank 0
        sample_count = gather_sum(sample_count)
        pre_pick_rank_1 = gather_sum(pre_pick_rank_1)
        pre_pick_rank_2 = gather_sum(pre_pick_rank_2)
        pre_pick_rank_3 = gather_sum(pre_pick_rank_3)
        post_pick_rank_1 = gather_sum(post_pick_rank_1)
        post_pick_rank_2 = gather_sum(post_pick_rank_2)
        post_pick_rank_3 = gather_sum(post_pick_rank_3)
        instance_pre_pick_rank_1 = gather_sum(instance_pre_pick_rank_1)
        instance_pre_pick_rank_2 = gather_sum(instance_pre_pick_rank_2)
        instance_pre_pick_rank_3 = gather_sum(instance_pre_pick_rank_3)
        instance_post_pick_rank_1 = gather_sum(instance_post_pick_rank_1)
        instance_post_pick_rank_2 = gather_sum(instance_post_pick_rank_2)
        instance_post_pick_rank_3 = gather_sum(instance_post_pick_rank_3)

        if torch.distributed.get_rank() != 0:
            return {}

    pre_pick_rank_1 /= sample_count
    pre_pick_rank_2 /= sample_count
    pre_pick_rank_3 /= sample_count
    post_pick_rank_1 /= sample_count
    post_pick_rank_2 /= sample_count
    post_pick_rank_3 /= sample_count
    instance_pre_pick_rank_1 /= sample_count
    instance_pre_pick_rank_2 /= sample_count
    instance_pre_pick_rank_3 /= sample_count
    instance_post_pick_rank_1 /= sample_count
    instance_post_pick_rank_2 /= sample_count
    instance_post_pick_rank_3 /= sample_count

    print("============= CENTROID ==================")
    print("Pre-pick:")
    print("Rank 1: ", pre_pick_rank_1)
    print("Rank 2: ", pre_pick_rank_2)
    print("Rank 3: ", pre_pick_rank_3)
    print("Post-pick:")
    print("Rank 1: ", post_pick_rank_1)
    print("Rank 2: ", post_pick_rank_2)
    print("Rank 3: ", post_pick_rank_3)
    print("============= INSTANCE ==================")
    print("Pre-pick:")
    print("Rank 1: ", instance_pre_pick_rank_1)
    print("Rank 2: ", instance_pre_pick_rank_2)
    print("Rank 3: ", instance_pre_pick_rank_3)
    print("Post-pick:")
    print("Rank 1: ", instance_post_pick_rank_1)
    print("Rank 2: ", instance_post_pick_rank_2)
    print("Rank 3: ", instance_post_pick_rank_3)

    return {
        'Pre@1': pre_pick_rank_1,
        'Pre@2': pre_pick_rank_2,
        'Pre@3': pre_pick_rank_3,
        'Post@1': post_pick_rank_1,
        'Post@2': post_pick_rank_1,
        'Post@3': post_pick_rank_1,
    }
