import csv

from demo_play import segmentation, logger
from demo_utils import *

flow_w = 768
flow_h = 512
ratio = flow_h / 1024


# %%


def width(img_frame, offset, od):
    if not od:
        img_frame = cv2.flip(img_frame, 1)

    img_frame = cv2.resize(img_frame, (flow_w, flow_h))
    seg_map = segmentation(img_frame)

    mirror_column = round(metric2pixel(offset, od) * ratio)
    pupilPts, _ = getValidPoints(seg_map[:, mirror_column:])
    pupil_ellipse = get_ellipse(pupilPts, seg_map)
    pupil_x = round(pupil_ellipse[0] + mirror_column)

    line = np.copy(seg_map[:, pupil_x]).astype('uint8')
    pts = np.where(line)
    rr = pts[0]
    if not rr.any():
        logger.write_silent('width: eyelid is not detected')
        return None, 0.0

    max_r, min_r = rr.max(), rr.min()
    mmPerPixel = mmPerPixel_R if od else mmPerPixel_L
    eye_width = round(((max_r - min_r) / ratio) * mmPerPixel, 1)

    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.line(img_gray, (mirror_column, 0), (mirror_column, flow_h), color=(0, 255, 0), thickness=1)
    cv2.line(img_gray, (pupil_x, min_r), (pupil_x, max_r), color=(0, 255, 0), thickness=1)

    '''repeat'''
    sig = img_frame.copy()
    cv2.line(sig, (mirror_column, 0), (mirror_column, flow_h), color=(0, 255, 0), thickness=2)
    im = np.uint8(255 * seg_map.astype(np.float32) / seg_map.max())
    edges = cv2.Canny(im, 50, 100) + cv2.Canny(255 - im, 50, 100)
    edges = cv2.dilate(edges, np.ones((3, 1), np.uint8), iterations=1)
    pts = np.where(edges)
    sig[pts] = [0, 255, 0]

    str_side = 'od'
    if not od:
        img_gray = cv2.flip(img_gray, 1)
        sig = cv2.flip(sig, 1)
        str_side = 'os'

    cv2.imwrite('log/' + str(int(time.time())) + '.jpg', img_gray)
    cv2.imwrite('log/widthSupport' + str_side + '.png', sig)
    with open('log/width.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['{}'.format('od' if od else 'os')] + [eye_width])

    return img_gray, eye_width, sig


if __name__ == '__main__':
    '''od'''
    # img_name = '110846/proptosis/134255preright.jpg'
    # frame = cv2.imread(img_name)
    # width_frame, width_value = width(frame, offset=13, od=True)
    '''os'''
    # img_name = '110846/proptosis/134246preleft.jpg'
    # frame = cv2.imread(img_name)
    # width_frame, width_value = width(frame, offset=0, od=False)

    img_name = 'check/widthOrProp_od.jpg'
    frame = cv2.imread(img_name)
    width_frame, width_value, sig = width(frame, offset=18.5, od=True)
    print(width_value)
