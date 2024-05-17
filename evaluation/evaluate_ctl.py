import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset

from dataset_armbench import ArmBenchDataset, test_armbench


def main():
    #seed = 1  # Fixing seed is not reallyt required for reproducibility. The ArmBench dataset is big enough to always get a consistent result.
    #torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description='Evaluate the pre-trained CTL model on ArmBench and BOP datasets')
    parser.add_argument('--pretrained_model_type', type=str, required=True, help="Type of the pre-trained model ['ViT-B_16', 'ResNet-50']")
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--armbench_test_batch_size', type=int, default=15, help='Batch size for ArmBench test')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loaders')
    parser.add_argument('--armbench_dataset_path', type=str, required=True, help='Path to the ArmBench dataset')
    args = parser.parse_args()

    test_kwargs = {'batch_size': args.armbench_test_batch_size,
                   'shuffle': False}
    cuda_kwargs = {'num_workers': args.num_workers,
                   'pin_memory': True}
    test_kwargs.update(cuda_kwargs)

    device = torch.device("cuda")

    armbench_test_dataset = ArmBenchDataset(args.armbench_dataset_path)

    armbench_test_loader = torch.utils.data.DataLoader(armbench_test_dataset, collate_fn=ArmBenchDataset.test_collate_fn, **test_kwargs)

    # load max model and weights
    if args.pretrained_model_type == 'ViT-B_16':
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    elif args.pretrained_model_type == 'ResNet-50':
        model = torchvision.models.resnet50(weights=None)
    weights = torch.load(args.pretrained_model_path)
    model.load_state_dict(weights)

    model = model.to(device)
    model = nn.DataParallel(model)

    # Run test before training
    print("=====================")
    print("Running test before training")
    test_armbench(device, model, armbench_test_loader, epoch=0)


if __name__ == '__main__':
    main()
