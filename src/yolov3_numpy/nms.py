"""Non-maximum suppression in NumPy."""
import numpy as np

from .config import size, yolo_iou_threshold, yolo_score_threshold


# reference: https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5

# @jit()s
def combined_non_max_suppression(boxes, scores, iou_threshold, score_threshold):
    # Return an empty list, if no boxes given
    if np.shape(boxes)[0] == 0:
        return []

    x1 = boxes[:, :, :, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, :, :, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, :, :, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, :, :, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    # `size` is the model input size from config
    # We add 1, because the pixel at the start as well as at the end counts
    areas = (x2 - x1 + 1/size) * (y2 - y1 + 1/size)
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.ones((np.shape(boxes)[0], np.shape(boxes)[1])) > 0
    classes = np.zeros((np.shape(boxes)[0], np.shape(boxes)[1]))
    for batch in range(np.shape(boxes)[0]):
        # indices[batch] = scores[batch,:] < score_threshold
        for i, box in enumerate(boxes[batch]):
            if np.all(scores[batch, i] < score_threshold):
                indices[batch, i] = False
            #   continue
        for i, box in enumerate(boxes[batch]):
            classes[batch, i] = np.where(
                scores[batch, i] == scores[batch, i].max())[0][0]
            scores[batch, i] = scores[batch, i].max()
            # Create temporary indices
            temp_indices = indices[batch].copy()
            temp_indices[i] = False
            # Find out the coordinates of the intersection box
            xx1 = np.maximum(box[0, 0], boxes[batch][temp_indices, 0, 0])
            yy1 = np.maximum(box[0, 1], boxes[batch][temp_indices, 0, 1])
            xx2 = np.minimum(box[0, 2], boxes[batch][temp_indices, 0, 2])
            yy2 = np.minimum(box[0, 3], boxes[batch][temp_indices, 0, 3])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1/size)
            h = np.maximum(0, yy2 - yy1 + 1/size)
            # compute the ratio of overlap
            overlap = (w * h) / areas[batch, temp_indices, 0]
            # print(overlap)

            # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
            if np.any(overlap > iou_threshold):
                indices[batch] = temp_indices

    # return only the boxes at the remaining indices
    boxes_result = boxes[indices]
    scores_result = scores[indices]
    classes_result = classes[indices]
    return boxes_result[:, 0, :], scores_result[:, 0], classes_result


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, objectness, class_probs
    b, c, t = [], [], []

    for o in outputs:
        b.append(np.reshape(o[0], (np.shape(o[0])[0], -1, np.shape(o[0])[-1])))
        c.append(np.reshape(o[1], (np.shape(o[1])[0], -1, np.shape(o[1])[-1])))
        t.append(np.reshape(o[2], (np.shape(o[2])[0], -1, np.shape(o[2])[-1])))

    bbox = np.concatenate(b, axis=1)
    confidence = np.concatenate(c, axis=1)
    class_probs = np.concatenate(t, axis=1)

    scores = confidence * class_probs

    boxes, scores, classes = combined_non_max_suppression(
        boxes=np.reshape(bbox, (np.shape(bbox)[0], -1, 1, 4)),
        scores=np.reshape(
            scores, (np.shape(scores)[0], -1, np.shape(scores)[-1])),
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return boxes, scores, classes
