"""Image pre/post-processing. Resizing is pure NumPy; drawing uses
OpenCV (color conversion), Pillow and seaborn (palette)."""
import numpy as np


# Reference: https://meghal-darji.medium.com/implementing-bilinear-interpolation-for-image-resizing-357cbb2c2722


def img_resize(original_img, new_h, new_w):
    import numpy as np
    import math
    # get dimensions of original image
    old_h, old_w, c = original_img.shape
    # create an array of the desired shape.
    # We will fill-in the values later.
    resized = np.zeros((new_h, new_w, c))
    # Calculate horizontal and vertical scaling factor
    w_scale_factor = (old_w) / (new_w) if new_h != 0 else 0
    h_scale_factor = (old_h) / (new_h) if new_w != 0 else 0
    for i in range(new_h):
        for j in range(new_w):
            # map the coordinates back to the original image
            x = i * h_scale_factor
            y = j * w_scale_factor
            # calculate the coordinate values for 4 surrounding pixels.
            x_floor = math.floor(x)
            x_ceil = min(old_h - 1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_w - 1, math.ceil(y))

            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_img[int(x), int(y), :]
            elif (x_ceil == x_floor):
                q1 = original_img[int(x), int(y_floor), :]
                q2 = original_img[int(x), int(y_ceil), :]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_img[int(x_floor), int(y), :]
                q2 = original_img[int(x_ceil), int(y), :]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
            else:
                v1 = original_img[x_floor, y_floor, :]
                v2 = original_img[x_ceil, y_floor, :]
                v3 = original_img[x_floor, y_ceil, :]
                v4 = original_img[x_ceil, y_ceil, :]

                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            resized[i, j, :] = q
    return resized.astype(np.uint8)


# Chuẩn hóa ảnh đầu vào về kích thước 416x416 và giá trị pixel trong khoản (0,1)
def transform_images(x_train, size):
    import cv2
    x_train = img_resize(x_train, size, size)
    x_train = x_train / 255
    return x_train


def draw_outputs(img, outputs, class_names):
    import cv2
    from seaborn import color_palette
    from PIL import Image, ImageDraw
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes = outputs
    # boxes, objectness, classes = boxes[0], objectness[0], classes[0]
    wh = np.flip(img.shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for i in range(len(boxes)):
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
        for t in np.linspace(0, 1, thickness):
            x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
            x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]],
                           outline=tuple(color))
        confidence = '{:.2f}%'.format(objectness[i]*100)
        text = '{} {}'.format(class_names[int(classes[i])], confidence)
        text_size = draw.textsize(text)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                       fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black')
    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img
