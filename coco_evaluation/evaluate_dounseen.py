import copy
import os
import json
import cv2
import numpy as np
import tqdm

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

import sys
dounseen_path = '/home/gouda/segmentation/image_agnostic_segmentation/scripts'
sys.path.insert(0, dounseen_path)
from dounseen.dounseen import UnseenSegment, UnseenClassifier, draw_segmented_image

from detectron2.evaluation import coco_evaluation
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import torch

# ___________________________________________________________________________________________________
# Parameters
# ___________________________________________________________________________________________________
#gt_json = '/media/gouda/ssd_data/datasets/dopose_coco/val/split_gt_coco_modal.json'
#gt_json = '/media/gouda/ssd_data/datasets/hope/val/split_gt_coco_modal.json'
gt_json = '/media/gouda/ssd_data/datasets/lm/test_all/split_gt_coco_modal.json'


#dataset_path = '/media/gouda/ssd_data/datasets/dopose_coco/val/'
#dataset_path = '/media/gouda/ssd_data/datasets/hope/val'
dataset_path = '/media/gouda/ssd_data/datasets/lm/test_all'

#gallery_path = '/media/gouda/ssd_data/datasets/dopose/classification/gallery/cropped/gallery_real_original_cropped_augmented_resized_256'
#gallery_path = '/media/gouda/ssd_data/datasets/hope/classification/gallery_real_resized_256_augmented'
gallery_path = '/media/gouda/ssd_data/datasets/lm/classification/gallery_synthetic_big_resized_256'

#camera_params = {'fx': 1778.81005859375, 'fy': 1778.870361328125, 'x_offset': 967.9315795898438, 'y_offset': 572.4088134765625}  # DoPose
#camera_params = {'fx': 1390.53, 'fy': 1386.99, 'x_offset': 964.957, 'y_offset': 522.586}  # HOPE
camera_params = {'fx': 572.4114, 'fy': 573.57043, 'x_offset': 325.2611, 'y_offset': 242.04899}  # LM

# dounseen parameters
seg_method = 'SAM'
classification_method = 'vit-b-16-ctl'
maskrcnn_model_path = os.path.join(dounseen_path, '../models/segmentation/segmentation_mask_rcnn.pth')
sam_model_path = os.path.join(dounseen_path, '../models/sam_vit_b_01ec64.pth')
classification_model_path = os.path.join(dounseen_path, '../models/classification/classification_vit_b_16_ctl.pth')
batch_size = 100

predictions_json = '/media/gouda/ssd_data/datasets/DoUnseen_evaluation/lm/scene_1_SAM.json'

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
    segmentor = UnseenSegment(method=seg_method, sam_model_path=sam_model_path, maskrcnn_model_path=maskrcnn_model_path)

classifier = UnseenClassifier(model_path=classification_model_path, gallery_images=gallery_path, gallery_buffered_path=None, augment_gallery=False, method=classification_method, batch_size=batch_size)

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
        # log time to segment image
        import datetime
        start_time = datetime.datetime.now()
        masks_boxes_predictions = segmentor.segment_image(rgb_img)
        end_time = datetime.datetime.now()
        print(f"Segmentation time: {end_time - start_time}")

        # log conversion time
        start_time = datetime.datetime.now()
        # convert predictions to detectron2 format
        boxes = Boxes(masks_boxes_predictions['bboxes'])
        # set scores all as 0.9
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

        annos = coco_evaluation.instances_to_coco_json(predictions['instances'].to('cpu'), image['id'])
        image_annotations.extend(annos)
        #from detectron2.data import MetadataCatalog
        #from detectron2.utils.visualizer import Visualizer
        #from detectron2.utils.visualizer import ColorMode
        #MetadataCatalog.get("user_data").set(thing_classes=[""])
        #metadata = MetadataCatalog.get("user_data")
        #v = Visualizer(rgb_img,
        #               metadata=metadata,
        #               instance_mode=ColorMode.IMAGE
        #               )
        #out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        #segmented_img = out.get_image()
        #cv2.imshow("img", segmented_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # time
        end_time = datetime.datetime.now()
        print(f"Conversion time: {end_time - start_time}")
        exit()
    # ===============================
    # Classify with MaskRCNN
    # ===============================
    #for id, annotation in enumerate(image_annotations):
    #    mask = gt_coco.annToMask(annotation)
    #    masked_rgb = cv2.bitwise_or(rgb_img,rgb_img,mask=mask)
    #    bbox = annotation['bbox']
    #    bbox = [int(val) for val in bbox]
    #    segment_img = masked_rgb[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    #    cv2.imshow("img", segment_img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    #    # Linemod dataset has one object annotated per image, so to evaluate linemod we need to search for that object
    #    # in this case the object to detect is the object in the coco annotation
    #    # search (find_object) of the annotation
    #    predicted_class = classifier.find_object(rgb_img, [mask], [bbox], obj_name=obj_name, centroid=False)
    #    predicted_class = int(predicted_class[-2:])
    #    #print(predicted_class)
    #    if predicted_class == annotation['category_id']:
    #        acc.append(True)
    #    else:
    #        acc.append(False)
    #    image_annotations[id]['category_id'] = predicted_class
    # ===============================

    # ===============================
    # Segment Linemod
    # ===============================
    query_images = []
    for id, annotation in enumerate(image_annotations):
        #polygon_rle = annotation['segmentation']['counts']
        mask = gt_coco.annToMask(annotation)
        masked_rgb = cv2.bitwise_or(rgb_img,rgb_img,mask=mask)
        bbox = annotation['bbox']
        bbox = [int(val) for val in bbox]
        segment_img = masked_rgb[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        query_images.append(segment_img)
        query_obj_id = [item for item in gt_data['annotations'] if item['image_id'] == image['id']][0]['category_id']
        image_annotations[id]['category_id'] = query_obj_id
    query_obj_name = 'obj_' + f'{query_obj_id:06}'
    segment_id = classifier.find_object(query_images, obj_name=query_obj_name, centroid=False)
    corresponding_anno = image_annotations[segment_id]
    #cv2.imshow("img", query_images[segment_id])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    predicted_annotations.append(image_annotations[segment_id])

#if seg_method == 'GT':
#    acc = np.array(acc)
#    print(np.where(acc == True)[0].shape[0]/acc.shape[0])
with open(predictions_json, 'w') as f:
    json.dump(predicted_annotations, f)
