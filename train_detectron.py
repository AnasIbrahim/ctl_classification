#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    #cfg.INPUT.MIN_SIZE_TRAIN = 50
    #cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 2  # DoPose 8 - HOPE 10

    training_dataset_path = '/media/gouda/ssd_data/datasets/hope/train_video'
    validation_dataset_path = '/media/gouda/ssd_data/datasets/hope/val'

    # DoPose
    #classes = [str(x+1) for x in range(0,18)] # BOP object count starts from 1
    # HOPE
    classes = [str(x+1) for x in range(0,28)] # BOP object count starts from 1

    #train_dataset_name = "dopose_train"
    train_dataset_name = "hope_video"
    register_coco_instances(train_dataset_name, {},
                            os.path.join(training_dataset_path, "split_gt_coco_modal.json"),
                            training_dataset_path)

    MetadataCatalog.get(train_dataset_name).set(thing_classes=classes)

    #val_dataset_name = "dopose_val"
    val_dataset_name = "hope_val"
    register_coco_instances(val_dataset_name, {},
                            os.path.join(validation_dataset_path, "split_gt_coco_modal.json"),
                            validation_dataset_path)
    MetadataCatalog.get(val_dataset_name).set(thing_classes=classes)

    cfg.DATASETS.TRAIN = [train_dataset_name]
    cfg.DATASETS.TEST = [val_dataset_name]
    cfg.TEST.EVAL_PERIOD = 1000

    # DoPose
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 18
    #cfg.MODEL.RETINANET.NUM_CLASSES = 18
    #cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 18
    # HOPE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    #cfg.MODEL.RETINANET.NUM_CLASSES = 28
    #cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 28

    cfg.SOLVER.MAX_ITER = 100000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.SOLVER.BASE_LR = 0.00025

    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    cfg.INPUT.RANDOM_FLIP = "none"

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    #cfg.MODEL.WEIGHTS = '/run/user/1002/gvfs/smb-share:server=129.217.152.32,share=data/Anas_Gouda/unseen_object_classification/train_dopose_detectron/train_1/model_0090499.pth'
    #cfg.MODEL.WEIGHTS = '/run/user/1002/gvfs/smb-share:server=129.217.152.32,share=data/Anas_Gouda/unseen_object_classification/train_hope_detectron/train_hope_video/model_0014999.pth'

    #cfg.OUTPUT_DIR = "/run/user/1002/gvfs/smb-share:server=129.217.152.32,share=data/Anas_Gouda/unseen_object_classification/train_dopose_detectron/train_X"
    cfg.OUTPUT_DIR = "/home/gouda/segmentation/ctl_training_output/scratch_training_output/detectron2/hope"
    #cfg.OUTPUT_DIR = "/run/user/1002/gvfs/smb-share:server=129.217.152.32,share=data/Anas_Gouda/unseen_object_classification/detectron2_evaluation"

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    args.resume = True
    args.num_gpus = 1
    args.eval_only = False
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
