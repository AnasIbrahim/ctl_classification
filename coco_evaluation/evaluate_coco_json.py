from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

#gt_json = '/media/gouda/ssd_data/datasets/dopose_coco/val/split_gt_coco_modal.json'
#gt_json = '/media/gouda/ssd_data/datasets/hope/val/split_gt_coco_modal.json'
gt_json = '/media/gouda/ssd_data/datasets/lm/test_all/split_gt_coco_modal.json'

# Results from different methods
# detectron2 trained on DoPose
#results_json = '/media/gouda/ssd_data/unseen_object_classification/train_dopose_detectron/train_1/inference/coco_instances_results.json'
#results_json = '/media/gouda/ssd_data/unseen_object_classification/train_hope_detectron/train_hope_video_3x/inference/coco_instances_results.json'
# GT masks + DoClassify
#results_json = '/media/gouda/ssd_data/unseen_object_classification/coco_evaluations/dopose/vit/doseg.json'
#results_json = '/media/gouda/ssd_data/unseen_object_classification/coco_evaluations/hope/vit/gt.json'
results_json = '/media/gouda/ssd_data/datasets/DoUnseen_evaluation/lm/scene_1_SAM.json'

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