# YOLOv3 from scratch (no TensorFlow / PyTorch)

A from-scratch implementation of the **YOLOv3 forward pass (inference)** written with
**NumPy** and accelerated with **Numba JIT**. No deep-learning framework is used by the model:
convolution, batch normalization, LeakyReLU, sigmoid, upsampling, box decoding and
non-maximum suppression are all hand-written.

It was originally built as a university project ("Applied Parallel Programming", HCMUS, 2022)
for **face mask detection** with 3 classes: `mask_weared_incorrect`, `with_mask`, `without_mask`,
trained on the [Kaggle Face Mask Detection dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
(853 images). This repository is a cleaned-up reorganization of
[TranPhu1999/Yolo3-face-mask-detection-from-scratch](https://github.com/TranPhu1999/Yolo3-face-mask-detection-from-scratch);
the algorithms were not rewritten.

## Dependencies

| Part | Dependency |
|---|---|
| Network layers, Darknet-53, YOLO heads, box decoding, NMS, bilinear resize | **NumPy** (+ `numba.jit` for speed) |
| Reading Darknet `.weights` | NumPy (`np.fromfile`) |
| Image read/write, BGR↔RGB conversion | OpenCV (`cv2`) |
| Drawing boxes | Pillow + seaborn (color palette) |
| `backends/cuda_numba.py` (experimental) | `numba.cuda` kernel for LeakyReLU → needs an NVIDIA GPU |
| `reference/yolov3_tensorflow.py` | **TensorFlow/Keras** — kept only as the baseline used to compare speed/results; not used by the package |

Other limitations:
* **Inference only.** There is no training code or loss function; the weights
  (`yolov3_mask.weights`, Darknet format) were trained elsewhere.
* Batch size 1, input fixed at 416×416.
* It is slow: the original report measured ~7 s (TensorFlow) vs ~160 s (NumPy + Numba JIT)
  vs >1 h (pure NumPy without JIT) per image on Colab.
* Known quirk preserved from the original: the upsampling step in `YoloConv` casts
  activations to `int` (`.astype(int)`), which may reduce accuracy vs the TF reference.

## Architecture

YOLOv3 = Darknet-53 backbone (52 conv layers with residual blocks) + 3 detection heads at
strides 32/16/8 (13×13, 26×26, 52×52 for a 416 input), 3 anchors per scale,
`3 × (5 + num_classes)` outputs per cell. Each `DarknetConv` = conv → batch norm → LeakyReLU(0.1).
Heads are joined with 2× nearest-neighbour upsampling + concatenation with backbone skip features.
Weights are streamed sequentially from the Darknet file while the forward pass runs.

## Install

```bash
git clone https://github.com/TranPhu1999/Yolov3-from-scratch-no-AI.git
cd Yolov3-from-scratch-no-AI
pip install -r requirements.txt
```

Python 3.8–3.11. `numpy<2`, `numba<0.59` and `Pillow<10` are pinned because the original code uses
`np.product`, `@jit()` object-mode fallback (in `Convolution_forward`) and `ImageDraw.textsize`.

## Weights & test image (not in this repo)

The trained weights (~235 MB) and a sample image are hosted on Google Drive (as in the original notebook):

```bash
pip install gdown
gdown --id 12hTWIBAWQXuDF4xpjsi1JDUlcYWRoowx   # yolov3_mask.weights
gdown --id 16H0NdwhOm-FFzar2rr0jETyKhXcGwamC   # maksssksksss0.png (sample image)
```

## Usage

```bash
# NumPy + Numba (CPU)
python scripts/detect.py -image maksssksksss0.png -weight yolov3_mask.weights
# -> writes jit_result_maksssksksss0.png

# Experimental CUDA variant (NVIDIA GPU)
PYTHONPATH=src python -m yolov3_numpy.backends.cuda_numba -image maksssksksss0.png -weight yolov3_mask.weights

# TensorFlow reference baseline (needs `pip install tensorflow absl-py`)
python reference/yolov3_tensorflow.py -image maksssksksss0.png -weight yolov3_mask.weights
```

From Python:

```python
import sys; sys.path.insert(0, "src")
from yolov3_numpy.weights import load_weights
from yolov3_numpy.detect import return_image
load_weights("yolov3_mask.weights")      # call before every image (weights are streamed)
annotated_bgr = return_image("maksssksksss0.png")
```

## Project structure

```
src/yolov3_numpy/
  config.py       anchors, thresholds, input size, class names
  layers.py       batch norm, correlate2d, convolution, LeakyReLU, sigmoid (NumPy + numba.jit)
  weights.py      Darknet .weights reader (sequential file handle)
  model.py        DarknetConv/Residual/Block, Darknet-53, YoloConv, YoloOutput, yolo_boxes, YoloV3
  nms.py          combined non-max suppression, yolo_nms
  image.py        bilinear resize (NumPy), preprocessing, box drawing
  detect.py       end-to-end detection on an image file
  backends/cuda_numba.py   experimental CUDA variant (self-contained)
scripts/detect.py          CLI
reference/yolov3_tensorflow.py   TensorFlow/Keras baseline (baseline only)
notebooks/Report_vi.ipynb  original project report (Vietnamese) with diagrams and timings
tests/test_sanity.py       layer/NMS/resize checks + full forward pass with random weights
```

Run tests: `pytest -q` (the full forward-pass test takes a few minutes on CPU).

## Credits

Authors: Trần Đức Phú, Lê Hữu Phúc, Lê Kinh Luân (HCMUS, 2022; instructor: Trần Trung Kiên).
References: [NMS](https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5),
[bilinear interpolation](https://meghal-darji.medium.com/implementing-bilinear-interpolation-for-image-resizing-357cbb2c2722).

## License

MIT
