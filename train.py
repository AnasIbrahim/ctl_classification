

import torch
import torch.distributed.elastic.multiprocessing.errors
import numpy as np
import torchvision.models
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
import subprocess

import wandb

from armbench_id import ARMBench, evaluate, Pick

@torch.distributed.elastic.multiprocessing.errors.record
def main(args):
    dataset = ARMBench(args.dataset, split='train', batch_size=args.batch_size, augment=args.augment)

    test_dataset = ARMBench(args.dataset, split='test', batch_size=args.batch_size)

    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)

    start_epoch = 0
    if args.continue_from:
        checkpoint = torch.load(args.continue_from, map_location='cpu')
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint['state'], 'module.')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch'] + 1

    if args.freeze_first_layer:
        first_layer = next(model.children())
        for param in first_layer.parameters():
            param.requires_grad = False

    if args.compile:
        model = torch.compile(model)

    is_distributed = ('LOCAL_RANK' in os.environ)
    if is_distributed:
        local_rank = int(os.environ['LOCAL_RANK'])

        print(f"torch.distributed setup (local_rank={local_rank})...")
        torch.distributed.init_process_group()

        device_id = local_rank
        torch.cuda.set_device(device_id)

        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        local_rank = None
        model = model.cuda()

    # In distributed mode, only local_rank == 0 should do output
    do_output = not is_distributed or local_rank == 0
    do_wandb = not is_distributed or (local_rank == 0 and torch.distributed.get_rank() == 0)

    test_view = test_dataset.test_view(subsample=20)
    test_loader = torch.utils.data.DataLoader(
        test_view,
        sampler=torch.utils.data.distributed.DistributedSampler(test_view) if is_distributed else None,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True
    )

    criterion = torch.nn.TripletMarginLoss(margin=0.1, p=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Obtain git status to include in the checkpoint
    git_commit = subprocess.run('git rev-parse HEAD', shell=True, stdout=subprocess.PIPE, check=True).stdout
    git_diff = subprocess.run('git diff', shell=True, stdout=subprocess.PIPE, check=True).stdout

    if do_wandb:
        wandb.init(project='object_identification', name=args.name, config={
            'git_commit': git_commit,
            'git_diff': git_diff,
            'args': vars(args)
        })

    if args.eval_first:
        metrics = evaluate(model, test_loader=test_loader, is_distributed=is_distributed, disable_progress=not do_output)
        if do_wandb:
            wandb.log(data = dict(
                epoch=start_epoch,
                **metrics
            ))

    for epoch in range(start_epoch, args.epochs):
        # Train!
        model.train()

        epoch_view = dataset.epoch_view()
        if is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(epoch_view)
        else:
            sampler = None

        loader = torch.utils.data.DataLoader(epoch_view, sampler=sampler, batch_size=None, num_workers=args.num_workers, pin_memory=True)

        for batch in tqdm(loader, disable=not do_output):
            images = batch['images'].cuda()
            agg_indices = batch['agg_indices'].cuda()
            agg_counts = batch['agg_counts'].cuda()
            image_index = batch['image_index'].cuda()

            optimizer.zero_grad()

            # Run all images through the network
            all_embeddings = model(images)

            # Normalize each embedding vector (TODO: necessary?)
            #all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=-1, p=2)

            # Add up all image embeddings in their aggregation slots
            aggregated = torch.zeros(agg_counts.shape[0], all_embeddings.shape[1], device=all_embeddings.device)
            aggregated.index_add_(dim=0, index=agg_indices, source=all_embeddings)

            # Divide by the count to compute the mean
            aggregated /= agg_counts.unsqueeze(1)

            # Now we want to get the anchor, positive, negative parts
            # Ignore aggregation slot 0, which is used for the unassigned images!
            anchor = aggregated[1::3]
            positive = aggregated[2::3]
            negative = aggregated[3::3]

            # With 50/50 chance, choose the single PickRGB query image instead of the aggregated queries
            if random.random() < 0.5:
                anchor = all_embeddings[image_index]

            loss = criterion(anchor, positive, negative)
            loss.backward()

            if do_wandb:
                wandb.log(data={
                    'loss': loss.item()
                })

            optimizer.step()

        # Evaluation
        model.eval()

        metrics = evaluate(model, test_loader=test_loader, is_distributed=is_distributed, disable_progress=not do_output)
        if do_wandb:
            wandb.log(data = dict(
                epoch=epoch + 1,
                **metrics
            ))

        # Checkpointing
        if do_wandb and ((epoch + 1) % args.checkpoint_every == 0):
            checkpoint = {
                'state': getattr(model, '_orig_mod', model).state_dict(),
                'epoch': epoch,
                'args': args,
                'git': {
                    'commit': git_commit,
                    'diff_to_head': git_diff,
                },
            }
            torch.save(checkpoint, f'checkpoints/{args.name}_epoch_{epoch}.model')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('high')

    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=Path, required=True,
        help='Path to ARMBench object ID dataset')
    parser.add_argument('--compile', action='store_true',
        help='Use torch.compile()')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument('--eval-first', action='store_true',
        help='Run evaluation before first epoch')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--freeze-first-layer', action='store_true')
    parser.add_argument('--name', type=str, required=True,
        help='Experiment name')
    parser.add_argument('--continue-from', type=Path,
        help='Continue from checkpoint')
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    main(args)
