import time
from pathlib import Path

import cv2
import numpy as np
from skimage.util import img_as_ubyte

from demo_play import segmentation, logger
from demo_utils import metric2pixel, getValidPoints, get_ellipse

flow_w = 768
flow_h = 512
ratio = flow_h / 1024
redness_offset = 5
redness_slope_multiplier = 1


# def redness(img_frame, path, od):
#     offset = 9
#
#     if not od:
#         img_frame = cv2.flip(img_frame, 1)
#
#     img_frame = cv2.resize(img_frame, (flow_w, flow_h))
#     seg_map = segmentation(img_frame)
#
#     mirror_column = round(metric2pixel(offset, od) * ratio)
#     filter_anti_sclera = seg_map != 1
#     if not np.any(filter_anti_sclera == False):
#         logger.write_silent('redness: sclera is not detected')
#         return None, 0.0
#
#     img_frame_uint8 = np.copy(img_frame).astype('uint8')
#     img_frame_uint8[filter_anti_sclera] = 0
#     img_frame_uint8[:, :mirror_column] = [0, 0, 0]
#
#     reg = LinearRegression().fit(img_frame_uint8[:, :, 2].reshape(-1, 1), img_frame_uint8[:, :, 0].reshape(-1, 1))
#     a = reg.coef_[0][0]
#     b = redness_offset
#     a1 = a * redness_slope_multiplier
#
#     filter1 = ((img_frame_uint8[:, :, 0] - a1 * img_frame_uint8[:, :, 2] + b) < 0) & (img_frame_uint8[:, :, 0] < 254) & (img_frame_uint8[:, :, 0] > 0)
#
#     sum_pixel_intensities = len(np.where(filter1)[0])
#     total_pixels = np.count_nonzero(cv2.cvtColor(img_frame_uint8, cv2.COLOR_BGR2GRAY))
#     redness_value = str(round((sum_pixel_intensities / total_pixels) * 100, 1)) + '%'
#
#     img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
#     img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
#     img_gray[filter1] = [255, 0, 0]
#     cv2.line(img_gray, (mirror_column, 0), (mirror_column, flow_h), color=(0, 255, 0), thickness=1)
#
#     if not od:
#         img_gray = cv2.flip(img_gray, 1)
#
#     cv2.imwrite(path, img_gray)
#
#     cv2.imwrite('log/' + str(int(time.time())) + '.jpg', img_gray)
#     with open('log/redness.csv', 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(['{}'.format('od' if od else 'os')] + [redness_value])
#
#     return img_gray, redness_value


def white_balance(img):
    white_patch_image = img_as_ubyte((img * 1.0 / np.percentile(img, 96, axis=(0, 1))).clip(0, 1))
    return white_patch_image


def ground_truth(image, img_patch=cv2.imread('pantone.jpg'), mode='mean'):
    if mode == 'mean':
        image_gt = ((image * (img_patch.mean() / image.mean(axis=(0, 1)))).clip(0, 255).astype(int))
    if mode == 'max':
        image_gt = ((image * 1.0 / img_patch.max(axis=(0, 1))).clip(0, 1))

    if image.shape[2] == 4:
        image_gt[:, :, 3] = 255

    return image_gt


def redness_quantification(img):
    B, G, R = cv2.split(img)
    B, G, R = np.mean(B), np.mean(G), np.mean(R)

    Max = np.maximum(np.maximum(B, G), R)
    Min = np.minimum(np.minimum(B, G), R)

    V = Max
    delta = Max - Min
    S = delta / V

    H = 0
    if delta != 0:
        if Max == R:
            H = (60 * ((G - B) / delta) + 360) / 360
        elif Max == G:
            H = (60 * ((B - R) / delta) + 360) / 360
        else:
            H = (60 * ((R - G) / delta) + 360) / 360

    score = H * S * 100

    return score


def redness(img_frame, offset=0, od=True, pts=[], testMode=False):
    """
    1. [pts] Example:
    [
        [100, 100],  # Point 1 (Size of image 1536x1024)
        [200, 50],   # Point 2
        [300, 100],  # Point 3
        [250, 200],  # Point 4
        [150, 200],  # Point 5
    ]

    2. Keep [testMode] False please
    """

    if not od:
        img_frame = cv2.flip(img_frame, 1)

    img_frame = cv2.resize(img_frame, (flow_w, flow_h))
    img_sclera = np.copy(img_frame).astype('uint8')

    if not pts:
        seg_map = segmentation(img_frame, True)
        if not np.any(seg_map == 1):
            logger.write_silent('redness: sclera is not detected')
            return None, 0.0

        mask = cv2.erode(seg_map.astype('uint8'), np.ones((16, 16), np.uint8), iterations=1)
        retval, labels = cv2.connectedComponents(mask)
        seg_map[labels != 1] = 0

        mirror_column = round(metric2pixel(offset, od) * ratio)
        img_sclera[:, :mirror_column] = [0, 0, 0]

        img_sclera[seg_map != 1] = [0, 0, 0]
        im = np.uint8(255 * seg_map.astype(np.float32) / seg_map.max())
        edges = cv2.Canny(im, 50, 100) + cv2.Canny(255 - im, 50, 100)
        main = np.where(edges)
    elif len(pts) > 2:
        img_sclera = np.copy(img_frame).astype('uint8')
        pts_int32 = np.array(pts, dtype=np.int32)
        pts_int32 = (pts_int32 * ratio).astype('int32')
        mask = np.zeros_like(img_frame[:, :, 0])
        cv2.fillPoly(mask, [pts_int32], (255, 255, 255))
        if not od:
            mask = cv2.flip(mask, 1)
        img_sclera[mask == 0] = [0, 0, 0]
        edges = cv2.Canny(mask, 50, 100) + cv2.Canny(255 - mask, 50, 100)
        main = np.where(edges)
    else:
        logger.write_silent('redness: at lease 3 points please')
        return None, 0.0

    # img_sclera = ground_truth(img_frame)
    score_redness = redness_quantification(img_sclera)
    score_redness = str(round(score_redness, 1))

    img_sclera = ground_truth(img_frame)
    img_sclera[main] = [0, 255, 0]

    if not od:
        img_sclera = cv2.flip(img_sclera, 1)

    if testMode:
        text = score_redness
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (00, 185)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        img_sclera = cv2.putText(img_sclera.astype(np.uint8), text, org, font, fontScale, color, thickness, cv2.LINE_AA,
                                 False)

    cv2.imwrite('log/' + str(int(time.time())) + '.jpg', img_sclera)

    return img_sclera, score_redness


if __name__ == '__main__':
    # for i in (Path.cwd() / 'samples_jpg').rglob('*preright.jpg'):
    #     img = cv2.imread(str(i))
    #     frame, score = redness(img, 'x.jpg', True)
    #     time.sleep(0.5)

    '''od'''
    # img_name = 'check/redness_od.bmp'
    # frame = cv2.imread(img_name)
    # redness_frame, redness_value = redness(frame, offset=12, od=True, pts=[
    #     [910, 390],  # Point 1
    #     [1060, 390],  # Point 2
    #     [1060, 490],  # Point 3
    #     [910, 490],  # Point 4
    # ])
    # print(redness_value)
    # redness_frame, redness_value = redness(frame, offset=12, od=True)
    # print(redness_value)

    '''os'''
    img_name = 'check/redness_od.bmp'
    frame = cv2.imread(img_name)
    # redness_frame, redness_value = redness(frame, offset=12, od=False, pts=[
    #     [452, 372],  # Point 1
    #     [744, 406],  # Point 2
    #     [760, 472],  # Point 3
    #     [639, 495],  # Point 4
    #     [450, 473],  # Point 4
    #     [470, 419],  # Point 4
    # ])
    # print(redness_value)
    redness_frame, redness_value = redness(frame, offset=12, od=True)
    print(redness_value)
