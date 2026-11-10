"""Sequential reader for Darknet `.weights` files.

The original code read weights layer by layer from a module-level open file
handle while building the forward pass; this module holds that handle.
"""
import numpy as np

weight_file = None


def load_weights(path):
    """Open a Darknet weights file and consume its 5-value header.

    Must be called before every forward pass (weights are streamed in order).
    """
    global weight_file
    if weight_file is not None:
        weight_file.close()
    weight_file = open(path, "rb")
    major, minor, revision, seen, _ = np.fromfile(
        weight_file, dtype=np.float32, count=5)
    return major, minor, revision, seen
