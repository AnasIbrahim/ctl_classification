# Learning Embeddings with Centroid Triplet Loss for Object Identification in Robotic Grasping
This repository contains the code for the paper "Learning Embeddings with Centroid Triplet Loss for Object Identification in Robotic Grasping"

[[Arxiv](https://arxiv.org/abs/2404.06277)]
[[Code](https://github.com/AnasIbrahim/ctl_classification)]

[Anas Gouda](https://github.com/AnasIbrahim),
[Max Schwarz](https://github.com/xqms),
[Christopher Reining](https://scholar.google.de/citations?user=cGwxRzAAAAAJ&hl=de),
[Sven Behnke](https://www.ais.uni-bonn.de/behnke/),
[Alice Kirchheim](https://scholar.google.de/citations?user=eY-wazgAAAAJ&hl=de),

This repository contains the training and evaluation code for the paper.
If you are only interested in using the pre-trained identification model or our whole pipeline in your project, you can refer to our [DoUnseen python package](https://github.com/AnasIbrahim/image_agnostic_segmentation).

## Installation

TODO

## Training

We use ViT model pre-trained on image-net.
Our final ViT model was trained on 8 NVIDIA A100 GPUs.
Install ViT pretrained model from PyTorch Hub:
```commandline
cd training
torchrun --standalone --nnodes=1 --nproc-per-node=8 train_ctl_model.py --dataset /PATH/TO/ARMBENCH_OBJECT_IDENT_DATASET --batch-size 128 --num-workers 4 --freeze-first-layer --name augment --lr 0.05 --epochs 200 --augment
```
To continue training from a checkpoint use the argument `--continue-from PATH/TO/CHECKPOINT.model`. The checkpoint contains the model weights, args and the last epoch number.

## Evaluation

### Evaluation on ARMBENCH object identification task
To evaluate our pretrained models on the ARMBENCH object identification task:
1. Make a new directory `models` in the root directory.
```commandline
mkdir models
```
2. Download our pre-trained model (ViT/ResNet) from [here](https://drive.google.com/drive/folders/10itUoEmgJAEN_cYkUOURc-3pTTll7rXv?usp=sharing) inside the `models` directory.
3. Run the following command to evaluate the model:
```commandline
cd evaluation
python evaluation/evaluate_ctl.py --pretrained_model_type 'ViT-B_16' --pretrained_model_path 'models/vit_b_16_ctl_augment_epoch_199.pth' --armbench_test_batch_size 15 --num_workers 8 --armbench_dataset_path /PATH/TO/ARMBENCH_OBJECT_IDENT_DATASET
```

## Citation
If you find this work useful, please consider citing:
```
@misc{gouda2024learning,
      title={Learning Embeddings with Centroid Triplet Loss for Object Identification in Robotic Grasping}, 
      author={Anas Gouda and Max Schwarz and Christopher Reining and Sven Behnke and Alice Kirchheim},
      year={2024},
      eprint={2404.06277},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```