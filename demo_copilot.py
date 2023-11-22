import cv2
import numpy as np

from demo_play import segmentation, logger
from demo_utils import metric2pixel, mmPerPixel_R, mmPerPixel_L

flow_w = 768
flow_h = 512
ratio = flow_h / 1024

motor_h = 450 * ratio  # TODO 卡点到照片顶部的像素距离


def copilot(img_frame, offset, od):
    if not od:
        img_frame = cv2.flip(img_frame, 1)

    img_frame = cv2.resize(img_frame, (flow_w, flow_h))
    seg_map = segmentation(img_frame)

    if len(list(np.unique(seg_map))) < 1:
        logger.write_silent('copilot: theres nothing')
        return 0, 0

    mirror_column = round(metric2pixel(offset, od) * ratio)
    mirror_row = int(motor_h)
    seg_map[:, :mirror_column] = 0
    rr, cc = np.where(seg_map)
    index = np.argmin(cc)
    eyelid_x = min(cc) - 55
    eyelid_y = rr[index] + 55

    pixel_togo_x = eyelid_x - mirror_column
    pixel_togo_y = eyelid_y - mirror_row

    mmPerPixel = mmPerPixel_R if od else mmPerPixel_L
    mm_togo_x = round(mmPerPixel * pixel_togo_x / ratio, 1)
    mm_togo_y = round(mmPerPixel * pixel_togo_y / ratio, 1)

    log = img_frame[:, :, 0].copy()
    cv2.line(log, (mirror_column, 0), (mirror_column, flow_h), color=128, thickness=1)
    cv2.line(log, (0, mirror_row), (flow_w, mirror_row), color=128, thickness=1)
    cv2.circle(log, (eyelid_x, eyelid_y), radius=9, color=255, thickness=-1)

    str_side = 'od'
    if not od:
        log = cv2.flip(log, 1)
        str_side = 'os'

    cv2.imwrite('log/copilot_' + str_side + '.jpg', log)

    return mm_togo_x, mm_togo_y


if __name__ == '__main__':
    '''od'''
    frame = cv2.imread('check/copilot/h_od.bmp')
    move_x1, move_y1 = copilot(frame, 0, od=True)
    '''os'''
    frame = cv2.imread('check/copilot/h_os.bmp')
    move_x2, move_y2 = copilot(frame, 0, od=False)

    move_x, move_y = (move_x1 + move_x2), (move_y1 + move_y2) / 2
    print(move_x1, move_x2, move_y1, move_y2)
    print('向内:', move_x)
    print('向上:', move_y)
