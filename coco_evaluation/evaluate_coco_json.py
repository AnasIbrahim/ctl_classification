from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

datasets_path = '/media/gouda/ssd_data/datasets'
dataset_name = 'hope'
dataset_path = os.path.join(datasets_path, dataset_name, 'val')
gt_json = os.path.join(dataset_path, 'split_gt_coco_modal.json')
results_json = '/home/gouda/segmentation/ctl_training_output/scratch_training_output/hope_eval/result.json'

annType = 'bbox'
#annType = 'segm'

# load gt_json

cocoGt=COCO(gt_json)
cocoDt=cocoGt.loadRes(results_json)

imgIds=sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()