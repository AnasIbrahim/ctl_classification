import argparse
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from dataset_armbench import ArmBenchDataset, test_armbench
from dataset_bop import BOPDataset, test_bop


def main():
    # TODO add argparse to set these parameters
    # pretrained model type
    # pretrained_model_type = 'ViT-B_16' or 'ResNet-50'
    #pretrained_model_path = '/home/gouda/segmentation/scratch_training_output/train_schwarz/augment_epoch_198.pth'
    #armbench_test_batch_size = 5 #6  # 15 ViT for 80GB GPU
    #bop_test_batch_size = 500
    #num_workers = 8
    #armbench_dataset_path = '/raid/datasets/armbench-object-id-0.1'
    #armbench_processed_data_file_test =  '/home/gouda/segmentation/ctl_classification/processed_armbench_dataset_test.pickle'
    #hope_dataset_path = '/raid/datasets/hope/classification'
    #output_path = '/home/gouda/segmentation/scratch_training_output/train_augmentation'

    #seed = 1  # Fixing seed is not required for reproducibility. The ArmBench dataset is big enough to always get a consistent result.
    #torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description='Evaluate the pre-trained CTL model on ArmBench and BOP datasets')
    parser.add_argument('--pretrained_model_type', type=str, required=True, help="Type of the pre-trained model ['ViT-B_16', 'ResNet-50']")
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--armbench_test_batch_size', type=int, default=5, help='Batch size for ArmBench test')
    parser.add_argument('--bop_test_batch_size', type=int, default=500, help='Batch size for BOP test')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loaders')
    parser.add_argument('--armbench_dataset_path', type=str, required=True, help='Path to the ArmBench dataset')
    parser.add_argument('--hope_dataset_path', type=str, required=True, help='Path to the BOP dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()
    

    test_kwargs = {'batch_size': args.sarmbench_test_batch_size,
                   'shuffle': False}
    cuda_kwargs = {'num_workers': args.num_workers,
                   'pin_memory': True}
    test_kwargs.update(cuda_kwargs)

    device = torch.device("cuda")

    if not os.path.exists(os.path.join(args.output_path, 'models')):
        os.makedirs(os.path.join(args.output_path, 'models'))

    armbench_test_dataset = ArmBenchDataset(args.armbench_dataset_path)

    bop_test_dataset = BOPDataset(args.hope_dataset_path)
    armbench_test_loader = torch.utils.data.DataLoader(armbench_test_dataset, collate_fn=ArmBenchDataset.test_collate_fn, **test_kwargs)
    bop_test_loader = torch.utils.data.DataLoader(bop_test_dataset, **test_kwargs)

    # load max model and weights
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    weights = torch.load(args.pretrained_model_path)
    model.load_state_dict(weights)

    model = model.to(device)
    model = nn.DataParallel(model)

    writer = SummaryWriter(os.path.join(args.output_path, 'logs'))

    # Run test before training
    print("=====================")
    print("Running test before training")
    test_bop(device, model, bop_test_loader, args.bop_test_batch_size, writer, epoch=0)
    test_armbench(device, model, armbench_test_loader, writer, epoch=0)


if __name__ == '__main__':
    main()
