"""Small sanity tests that need no weights file."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yolov3_numpy.layers import (BatchNormalization_forward, correlate2d,  # noqa: E402
                                 Convolution_forward, npLeakyReLU, sigmoid)
from yolov3_numpy.image import img_resize  # noqa: E402
from yolov3_numpy.nms import combined_non_max_suppression  # noqa: E402
from yolov3_numpy import model, weights  # noqa: E402


def test_correlate2d_matches_naive():
    rng = np.random.default_rng(0)
    a, k = rng.random((6, 6)), rng.random((3, 3))
    out = correlate2d(a, k, 1, "valid")
    ref = np.array([[np.sum(a[i:i+3, j:j+3] * k) for j in range(4)] for i in range(4)])
    assert np.allclose(out, ref)
    assert correlate2d(a, k, 1, "same").shape == (6, 6)


def test_convolution_shapes():
    x = np.random.rand(1, 8, 8, 2)
    k = np.random.rand(3, 3, 2, 4)
    y = Convolution_forward(x, k, 4, True, np.zeros(4), 1, "same")
    assert y.shape == (1, 8, 8, 4)


def test_activations_and_bn():
    x = np.array([[[[-1.0, 2.0]]]])
    assert np.allclose(npLeakyReLU(x.copy(), 0.1), [[[[-0.1, 2.0]]]])
    s = sigmoid(np.zeros((1, 2, 2, 1, 1)))
    assert np.allclose(s, 0.5)
    bn = BatchNormalization_forward(np.ones((1, 1, 1, 2)), np.ones(2), np.zeros(2),
                                    np.ones(2), np.ones(2) - 0.001)
    assert np.allclose(bn, 0)


def test_img_resize():
    img = (np.random.rand(20, 30, 3) * 255).astype(np.uint8)
    assert img_resize(img, 10, 12).shape == (10, 12, 3)


def test_nms_removes_overlap():
    boxes = np.array([[[[0.1, 0.1, 0.5, 0.5]], [[0.11, 0.11, 0.5, 0.5]], [[0.6, 0.6, 0.9, 0.9]]]])
    scores = np.array([[[0.9, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.7, 0.0]]])
    b, s, c = combined_non_max_suppression(boxes, scores, 0.5, 0.5)
    assert len(b) == 2 and list(c) == [0, 1]


def test_full_forward_with_random_weights(monkeypatch):
    """Build the whole network on a tiny 32x32 input with random weights
    (np.fromfile is stubbed so no 240 MB weights file is needed)."""
    rng = np.random.default_rng(0)
    monkeypatch.setattr(weights, "weight_file", object())
    monkeypatch.setattr(np, "fromfile",
                        lambda f, dtype, count: (rng.random(count) * 0.01 + 0.001).astype(dtype))
    boxes, scores, classes = model.YoloV3(np.random.rand(1, 32, 32, 3), classes=3)
    assert len(boxes) == len(scores) == len(classes)
