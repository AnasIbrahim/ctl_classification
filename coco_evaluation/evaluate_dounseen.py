import copy
import os
import json
import cv2
import numpy as np
import tqdm

import torch

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

from detectron2.evaluation import coco_evaluation
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes



import sys
dounseen_path = '/home/gouda/segmentation/image_agnostic_segmentation/scripts'
sys.path.insert(0, dounseen_path)
from dounseen.dounseen import UnseenSegment, UnseenClassifier
from dounseen import utils

# ___________________________________________________________________________________________________
# Parameters
# ___________________________________________________________________________________________________
datasets_path = '/media/gouda/ssd_data/datasets'
predictions_json = '/home/gouda/segmentation/ctl_training_output/scratch_training_output/hope_eval/result.json'

#lm_query_obj_name = 'obj_000008'

dataset_name = 'hope'
dataset_path = os.path.join(datasets_path, dataset_name, 'val')
gt_json = os.path.join(dataset_path, 'split_gt_coco_modal.json')
gallery_path = os.path.join(datasets_path, dataset_name,'classification/gallery_real_resized_256_augmented')

# dounseen parameters
seg_method = 'GT'
classification_method = 'vit-b-16-ctl'
maskrcnn_model_path = os.path.join(dounseen_path, '../models/segmentation/segmentation_mask_rcnn.pth')
sam_model_path = os.path.join(dounseen_path, '../models/sam_vit_b_01ec64.pth')
classification_model_path = os.path.join(dounseen_path, '../models/classification/classification_vit_b_16_ctl.pth')
classification_batch_size = 100
# ___________________________________________________________________________________________________

segmentation_methods = ['GT', 'maskrcnn', 'SAM']
classification_methods = ['resnet-50-ctl', 'vit-b-16-ctl']
device = 'cuda'

maskrcnn_model_path = os.path.join(dounseen_path, '../models/segmentation/segmentation_mask_rcnn.pth')
sam_model_path = os.path.join(dounseen_path, '../models/sam_vit_b_01ec64.pth')

gt_data = json.load(open(gt_json))
gt_coco = COCO(gt_json)

# make DoUnseen segmentation and classification objects
if seg_method != 'GT':
    segmentor = UnseenSegment(method=seg_method, sam_model_path=sam_model_path, maskrcnn_model_path=maskrcnn_model_path, filter_sam_predictions=True, smallest_segment_size=50*50)

classifier = UnseenClassifier(model_path=classification_model_path, gallery_images=gallery_path, gallery_buffered_path=None, augment_gallery=False, method=classification_method, batch_size=classification_batch_size)

last_obj_name = None

anno_id_count = 0
acc = []
predicted_annotations = []
images = gt_data['images']
for image in tqdm.tqdm(images):
    rgb_img_path = os.path.join(dataset_path,image['file_name'])
    rgb_img = cv2.imread(rgb_img_path)
    # ===============================
    # Segmentation
    # ===============================
    base_anno = {'id': None, 'image_id': image['id'], 'category_id': None, 'iscrowd': 0, 'area': None, 'bbox': None,
                 'segmentation': {'counts': None, 'size': [1200, 1944]}, 'width': 1944, 'height': 1200, 'ignore': False, 'score': None}
    image_annotations = []
    # GT masks
    if seg_method == 'GT':
        image_annotations = [item for item in gt_data['annotations'] if item['image_id'] == image['id']]
        for anno in image_annotations:
            #anno.update({'category_id': None})  # will be used for evaluation to check if classification is correct then replaced after the classification prediction
            anno['score'] = 1.00  # all segmentation from GT are correct
    # DoSegment
    elif seg_method == 'maskrcnn' or seg_method == 'SAM':
        masks_boxes_predictions = segmentor.segment_image(rgb_img)

        # convert predictions to detectron2 format
        boxes = Boxes(masks_boxes_predictions['bboxes'])
        # set scores all as 0.9 - this is dummy and not being used anywhere
        scores = torch.tensor([0.9] * len(masks_boxes_predictions['bboxes']))
        # set all labels as 0
        labels = torch.tensor([0] * len(masks_boxes_predictions['bboxes']))
        masks = masks_boxes_predictions['masks']
        masks = np.array(masks)
        masks = torch.tensor(masks, dtype=torch.uint8)
        instances = Instances(rgb_img.shape[:2])
        instances.set('pred_boxes', boxes)
        instances.set('scores', scores)
        instances.set('pred_classes', labels)
        instances.set('pred_masks', masks)
        predictions = {'instances': instances}

        image_annotations = coco_evaluation.instances_to_coco_json(predictions['instances'].to('cpu'), image['id'])

        # visualization
        from detectron2.data import MetadataCatalog
        from detectron2.utils.visualizer import Visualizer
        from detectron2.utils.visualizer import ColorMode
        MetadataCatalog.get("user_data").set(thing_classes=[""])
        metadata = MetadataCatalog.get("user_data")
        v = Visualizer(rgb_img,
                       metadata=metadata,
                       instance_mode=ColorMode.IMAGE
                       )
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        segmented_img = out.get_image()
        # show image resized to half
        cv2.imshow("img", cv2.resize(segmented_img, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ===============================
    # Classify with MaskRCNN
    # ===============================
    segments = []
    for id, annotation in enumerate(image_annotations):
        mask = gt_coco.annToMask(annotation)
        # TODO make background white
        masked_rgb = cv2.bitwise_or(rgb_img,rgb_img,mask=mask)
        bbox = annotation['bbox']
        bbox = [int(val) for val in bbox]
        segment_img = masked_rgb[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        #cv2.imshow("img", segment_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        segments.append(segment_img)
    predicted_classes, predicted_scores = classifier.classify_all_objects(segments, threshold=0.0)
    predicted_classes = predicted_classes.tolist()
    predicted_scores = predicted_scores.tolist()
    # remove predicted_classes with class -1 and their corresponding annotations
    new_annotations = []
    for idx in range(len(predicted_classes)):
        if predicted_classes[idx] != -1:
            image_annotations[idx]['category_id'] = predicted_classes[idx] + 1
            image_annotations[idx]['score'] = predicted_scores[idx]
            new_annotations.append(image_annotations[idx])
    predicted_annotations.extend(new_annotations)
    # ===============================

    # ===============================
    # Segment Linemod
    # ===============================
    #query_images = []
    #for id, annotation in enumerate(image_annotations):
    #    #polygon_rle = annotation['segmentation']['counts']
    #    mask = gt_coco.annToMask(annotation)
    #    # apply mask to rgb image but make the background white
    #    masked_rgb = copy.deepcopy(rgb_img)
    #    masked_rgb[mask == 0] = [255, 255, 255]
    #    bbox = annotation['bbox']
    #    bbox = [int(val) for val in bbox]
    #    segment_img = masked_rgb[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    #    query_images.append(segment_img)
    #    query_obj_id = [item for item in gt_data['annotations'] if item['image_id'] == image['id']][0]['category_id']
    #    image_annotations[id]['category_id'] = query_obj_id
    #if lm_query_obj_name is None:  # if no object name is given, use the object id
    #    lm_query_obj_name = 'obj_' + f'{query_obj_id:06}'
    #segment_id, score = classifier.find_object(query_images, obj_name=lm_query_obj_name, method='max')
    #print("score: " + str(score))
    #corresponding_anno = image_annotations[segment_id]
    ## save segmented image
    ##cv2.imwrite(os.path.join(save_img_path, image['file_name']), query_images[segment_id])
    #cv2.imshow("img", cv2.resize(query_images[segment_id], (0, 0), fx=0.5, fy=0.5))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #predicted_annotations.append(image_annotations[segment_id])

#if seg_method == 'GT':
#    acc = np.array(acc)
#    print(np.where(acc == True)[0].shape[0]/acc.shape[0])
with open(predictions_json, 'w') as f:
    json.dump(predicted_annotations, f)
