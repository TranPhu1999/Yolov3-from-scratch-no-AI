#!/usr/bin/env python
"""Detect face masks in an image with the NumPy YOLOv3 implementation.

Usage: python scripts/detect.py -image path/to/img.png -weight yolov3_mask.weights
Writes `jit_result_<image name>` (same as the original faceMaskDetection.py).
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")

import cv2  # noqa: E402

from yolov3_numpy.weights import load_weights  # noqa: E402
from yolov3_numpy.detect import return_image  # noqa: E402


def main():
    all_args = argparse.ArgumentParser()
    all_args.add_argument("-image", "--path_to_img", required=True)
    all_args.add_argument("-weight", "--path_to_weight", required=True)
    all_args.add_argument("-o", "--output", default=None,
                          help="output path (default: jit_result_<image name>)")
    args = vars(all_args.parse_args())

    load_weights(args["path_to_weight"])

    t1 = time.time()
    img = return_image(args["path_to_img"])
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    out = args["output"] or "jit_result_" + args["path_to_img"].split("/")[-1]
    cv2.imwrite(out, img)


if __name__ == "__main__":
    main()
