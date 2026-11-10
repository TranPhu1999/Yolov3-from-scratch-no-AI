"""Low-level NumPy layers (Numba-JIT compiled): batch norm, 2D correlation,
convolution, LeakyReLU and sigmoid."""
import numpy as np
from numba import jit


@jit(nopython=True)
def BatchNormalization_forward(input, gamma, beta, moving_mean, moving_variance, epsilon=0.001):

    mean_x = moving_mean.copy()
    var_x = moving_variance.copy()

    var_x += epsilon
    stddev_x = np.sqrt(var_x)
    x_minus_mean = input - mean_x
    standard_x = x_minus_mean / stddev_x
    return gamma * standard_x + beta


# Phép correlate tham khảo từ đây https://numpy.org/doc/stable/reference/generated/numpy.convolve.html,
# https://docs.scipy.org/doc//scipy-1.3.0/reference/generated/scipy.signal.correlate2d.html
@jit(nopython=True)
def correlate2d(input, kernel, stride=1, padding="valid"):
    h_i, w_i = input.shape
    h_k, w_k = kernel.shape

    s_h = stride
    s_w = stride

    if padding == 'valid':
        p_h = 0
        p_w = 0

    if padding == 'same':
        p_h = int((h_k - 1)/2)
        p_w = int((w_k - 1)/2)

    behind = np.zeros((h_i+2*p_h, w_i+2*p_w))
    behind[p_h:h_i+p_h, p_w:w_i+p_w] = input
    input = behind

    h_out = int((h_i - h_k + 2*p_h)/stride + 1)
    w_out = int((w_i - w_k + 2*p_w)/stride + 1)

    output_conv = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            for i_kernel in range(h_k):
                for j_kernel in range(w_k):
                    output_conv[i, j] += input[i_kernel+i*stride,
                                               j_kernel+j*stride]*kernel[i_kernel, j_kernel]

    return output_conv

# Tích chập tiến
# @jit(nopython=True)

@jit()
def Convolution_forward(input, kernel, filters, use_batchnorm=True, bias=[[[]]], stride=1, padding="valid"):

    _, input_height, input_width, input_depth = input.shape
    kernel_height, kernel_witdh, _, _ = kernel.shape
    # input shape is [batch_size, height, width, input_depth]
    # kernel shape is [kernel_size,kernel_size,input_depth,filters]
    # output shape is [height, width, filters]

    output_shape = (int(input_height/stride), int(input_width/stride), filters)
    output = np.zeros(output_shape)

    if use_batchnorm == False:
        for i in range(filters):
            output[:, :, i] += bias[i]
    # if use_batchnorm == False:
    #   for h in range(len(output)):
    #     for k in range(len(output[h])):
    #       for i in range(filters):
    #         output[h,k,i] += bias[i]
    for i in range(filters):
        temp = np.zeros(output_shape[:-1])
        for j in range(input_depth):
            temp += correlate2d(input[0, :, :, j], kernel[:,
                                :, j, i], stride=stride, padding=padding)
        output[:, :, i] += temp

    output = np.expand_dims(output, 0)
    return output

# hàm kích hoạt leakyReLU


@jit(nopython=True)
def npLeakyReLU(x, alpha=0.01):
    (n, x_h, x_w, n_ker) = x.shape
    for k in range(n_ker):
        for r in range(x_h):
            for c in range(x_w):
                if x[0, r, c, k] < 0:
                    x[0, r, c, k] = alpha*x[0, r, c, k]
    return x


# Output của YOLO có lưu xác xuất bounding box thuộc các class
# (ví dụ như có 3 class thì sẽ có một list độ dài 3 lưu xác xuất box đó có thuộc class đó không)
# vậy nên cần dùng sigmoid để trả về giá trị từ (0-1)


# @jit(nopython=True)
# def sigmoid(x):
#     z = np.exp(-x)
#     sig = 1 / (1 + z)
#     return sig

@jit(nopython=True)
def sigmoid(x):
    (n, s, y, z, t) = np.shape(x)
    for i in range(s):
        for j in range(y):
            for k in range(z):
                for h in range(t):
                    temp = np.exp(-x[0, i, j, k, h])
                    x[0, i, j, k, h] = 1 / (1 + temp)
    return x
