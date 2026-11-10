"""Model / inference configuration (from the original faceMaskDetection.py)."""
import numpy as np

yolo_max_boxes = 10
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

size = 416                 # size images are resized to for model
num_classes = 3            # number of classes in model
class_names = ["mask_weared_incorrect,", "with_mask", "without_mask"]
