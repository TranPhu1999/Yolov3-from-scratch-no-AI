"""End-to-end detection on an image file."""
import cv2
import numpy as np

from .config import size, num_classes, class_names
from .image import transform_images, draw_outputs
from .model import YoloV3


def return_image(filename):
  """Run detection on an image file; returns the annotated BGR image.
  Call weights.load_weights(path) first."""
  img_raw = cv2.imread(filename)
  img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

  img = transform_images(img_raw, size)
  img = np.expand_dims(img, 0)

  boxes, scores, classes = YoloV3(img, classes=num_classes)

  print('detections:')

  class_names_local = class_names

  for i in range(len(boxes)):
      print('\t{}, {}, {}'.format(class_names_local[int(
          classes[i])], np.array(scores[i]), np.array(boxes[i])))
  img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
  img = draw_outputs(img, (boxes, scores, classes), class_names_local)

  return img
