"""YOLOv3 (Darknet-53 backbone + 3-scale head) forward pass in NumPy."""
import time

import numpy as np

from . import weights as _weights
from .config import yolo_anchors, yolo_anchor_masks
from .layers import (BatchNormalization_forward, Convolution_forward,
                     npLeakyReLU, sigmoid)
from .nms import yolo_nms


# Layer Darknet Conv bao gồm 1 layer convole đi kèm với batch normalization và leakyReLU

def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        print(x.shape)
        x = np.pad(x, ((0, 0), (1, 0), (1, 0), (0, 0)),
                   'constant')  # top left half-padding
        padding = 'valid'

# Load weight------------------------------------------------------------

    weight_file = _weights.weight_file

    bias = None
    if batch_norm is False:
        # read bias weight of convolutional layer if there is no batch normalization
        bias = np.fromfile(weight_file, dtype=np.float32, count=filters)
    else:
        # read batch normalization layer weight
        bn_weights = np.fromfile(
            weight_file, dtype=np.float32, count=4*filters)
        bn_weights = bn_weights.reshape((4, filters))
        beta, gamma, moving_mean, moving_variance = bn_weights

    # read kernel weight
    conv_shape = (filters, x.shape[-1], size, size)
    kernel = np.fromfile(weight_file, dtype=np.float32,
                         count=np.product(conv_shape))
    kernel = kernel.reshape(conv_shape).transpose([2, 3, 1, 0])
# ------------------------------------------------------------------------
    x = Convolution_forward(input=x, kernel=kernel, filters=filters,
                            bias=bias, stride=strides, padding=padding, use_batchnorm=batch_norm)
    if batch_norm:
        x = BatchNormalization_forward(
            x, gamma, beta, moving_mean, moving_variance)
        x = npLeakyReLU(x, alpha=0.1)
    return x


def DarknetResidual(x, filters):
    prev = x  # Skip connection, giúp các mạng neural có cấu trúc quá sâu giảm thiểu mất mát feature khi đi xuống
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = prev + x
    return x

# Mỗi Darknet Block gồm 1 Darknet convole và n Darknet Residual


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(inputs):
    x = inputs
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return x_36, x_61, x

# Block các layer riêng của YOLO dùng cho object detection


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            x, x_skip = x_in

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = np.kron(x, np.ones((2, 1))).repeat(2, axis=1).astype(int)
            # x = Concatenate()([x, x_skip])
            x = np.concatenate((x, x_skip), axis=3)
        else:
            x = x_in

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return x
    return yolo_conv

# Layer trả output


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = x_in
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = (lambda x: np.reshape(x, (-1, np.shape(x)[1], np.shape(x)[2],
                                      anchors, classes + 5)))(x)
        return x
    return yolo_output



def yolo_boxes(pred, anchors, classes):
    grid_size = np.shape(pred)[1]

    box_xy, box_wh, objectness, class_probs = np.split(
        pred, (2, 4, 5), axis=-1)

    box_xy = sigmoid(box_xy)
    objectness = sigmoid(objectness)
    class_probs = sigmoid(class_probs)

    pred_box = np.concatenate((box_xy, box_wh), axis=-1)

    grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    grid = np.expand_dims(np.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + grid.astype(np.float32)) / float(grid_size)
    box_wh = np.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = np.concatenate([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box



def YoloV3(inputs, size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=3):

    import time
    start1 = time.time()

    x = inputs
    x_36, x_61, x = Darknet(x)

    x = YoloConv(512)(x)
    output_0 = YoloOutput(512, len(masks[0]), classes)(x)

    x = YoloConv(256)((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes)(x)

    x = YoloConv(128)((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes)(x)

    end1 = time.time()
    print("Conv: ", end1 - start1)

    boxes_0 = (lambda x: yolo_boxes(x, anchors[masks[0]], classes))(output_0)
    boxes_1 = (lambda x: yolo_boxes(x, anchors[masks[1]], classes))(output_1)
    boxes_2 = (lambda x: yolo_boxes(x, anchors[masks[2]], classes))(output_2)
    end2 = time.time()
    print("Yolo Box: ", end2-end1)
    outputs = (lambda x: yolo_nms(x, anchors, masks, classes))(
        (boxes_0[:3], boxes_1[:3], boxes_2[:3]))
    end3 = time.time()
    print("Non-max supperession: ", end3-end2)

    return outputs
