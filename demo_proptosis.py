import csv
from math import pi, atan2, tan

from demo_play import segmentation, logger
from demo_utils import *

flow_w = 1536
flow_h = 1024
ratio = flow_h / 1024
pixel_size = 0.0048

flange_focal = 17.5


# %%


def adjust_gamma(img_gray, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_gray, table)


def find_reflection_point(image, seg_map, mirror_column, threshold=10, N=50):
    seg_map_uint8 = np.copy(seg_map[:, mirror_column:]).astype('uint8')
    seg_map_uint8[seg_map_uint8 == 2] = 1
    rr, cc = np.where(seg_map_uint8 == 1)
    min_r, max_r = min(rr), max(rr) + 20
    crop = image[:, :mirror_column]
    # cv2.imwrite('crop.png', crop)

    gamma = adjust_gamma(crop, gamma=0.1)
    mask = (gamma >= threshold).astype('uint8')
    mask = cv2.dilate(mask, np.ones((20, 5), np.uint8), iterations=1)
    mask = cv2.erode(mask, np.ones((16, 4), np.uint8), iterations=1)

    retval, labels = cv2.connectedComponents(mask)

    label_list = []
    num = labels.max()
    for i in range(1, num + 1):
        pts = np.where(labels == i)
        if len(pts[0]) < N or pts[0].max() > max_r or pts[0].min() < min_r:
            labels[pts] = 0
        else:
            label_list.append(pts)

    main_copy = cv2.normalize(src=labels, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    main_copy = cv2.applyColorMap(main_copy, 16)
    cv2.imwrite('a.png', main_copy)
    cv2.imwrite('b.png', gamma)

    max_height = 0
    main = ()
    for label in label_list:
        height = label[0].max() - label[0].min()
        if max_height < height:
            max_height = height
            main = label

    main = (main[0], main[1])

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image[:, :, 0][main] = 0
    image[:, :, 1][main] = 0
    image[:, :, 2][main] = 255
    image[:, :, 2][max_r, :] = 255

    if main:
        depth = main[1].min()
    else:
        depth = 0

    return depth, image


def find_image_angle(point1, point2, focal, pixel=pixel_size):
    return atan2(abs(point1 - point2) * pixel, focal)


def find_depth_ratio(alpha, beta, a, b):
    y = (b * tan(beta) - a) / (tan(beta) + tan(pi / 2.0 - alpha))
    x = b - y
    return x, y


def find_offset_recursive(j, gamma, delta, threshold=0.001):
    e = 0
    f = (j - e) * tan(delta)
    temp = f * tan(gamma)
    while abs(temp - e) > threshold:
        e = temp
        f = (j - e) * tan(delta)
        temp = f * tan(gamma)
    return f


def compute(pupil, reflection, alpha, focal, depth, g, mirror_offset, principal=flow_w / 2):
    h = flange_focal - focal + depth
    beta = find_image_angle(reflection, principal, focal)
    gamma = find_image_angle(principal, pupil, focal)
    delta = pi / 2.0 + beta - 2.0 * alpha
    a, b = find_depth_ratio(alpha, beta, g, h)
    c = a * tan(beta)
    i = a * tan(gamma)
    f = find_offset_recursive(c + i, gamma, delta)
    result = b + f + mirror_offset
    return result


def proptosis(img_frame, offset, od):
    if not od:
        img_frame = cv2.flip(img_frame, 1)

    seg_map = segmentation(img_frame)
    seg_map = cv2.resize(seg_map, (1536, 1024), interpolation=cv2.INTER_NEAREST)
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    if od:
        pixelPerMm = pixelPerMm_R
        mirror_offset = mirror_offset_R

        depth = depth_R
        focal_length = focal_length_R
        center_position = center_position_R
        degree = degree_R
    else:
        pixelPerMm = pixelPerMm_L
        mirror_offset = mirror_offset_L

        depth = depth_L
        focal_length = focal_length_L
        center_position = center_position_L
        degree = degree_L

    pupilPts, _ = getValidPoints(seg_map)
    pupil_ellipse = get_ellipse(pupilPts, seg_map)
    if pupil_ellipse is None:
        logger.write_silent('proptosis: ellipse cannot be fitted properly')
        return 0.0, None
    pupil_x, pupil_y = round(pupil_ellipse[0]), round(pupil_ellipse[1])

    mirror_column = round(metric2pixel(offset, od) * ratio)
    mirror_column_bias = round(mirror_column - (pixelPerMm * mirror_offset) * ratio)
    reflection, reflected_eye = find_reflection_point(img_gray, seg_map, mirror_column_bias)

    '''draw'''
    cv2.line(reflected_eye, (pupil_x, 0), (pupil_x, flow_h), color=(0, 255, 0), thickness=2)
    cv2.line(reflected_eye, (int(flow_w / 2), 0), (int(flow_w / 2), flow_h), color=(0, 255, 0), thickness=2)
    cv2.line(reflected_eye, (mirror_column, 0), (mirror_column, flow_h), color=(0, 255, 0), thickness=2)
    cv2.line(reflected_eye, (round(reflection), 0), (round(reflection), flow_h), color=(0, 255, 0), thickness=2)

    cv2.circle(reflected_eye, (pupil_x, pupil_y), radius=9, color=(0, 0, 255), thickness=-1)

    prop = compute(pupil_x, reflection, degree * pi / 180, focal_length, depth, center_position - offset, mirror_offset)

    '''repeat'''
    sig = img_frame.copy()
    cv2.line(sig, (round(reflection), 0), (round(reflection), flow_h), color=(0, 255, 0), thickness=2)
    cv2.line(sig, (mirror_column, 0), (mirror_column, flow_h), color=(0, 255, 0), thickness=2)
    im = np.uint8(255 * seg_map.astype(np.float32) / seg_map.max())
    edges = cv2.Canny(im, 50, 100) + cv2.Canny(255 - im, 50, 100)
    edges = cv2.dilate(edges, np.ones((3, 1), np.uint8), iterations=1)
    pts = np.where(edges)
    sig[pts] = [0, 255, 0]

    str_side = 'od'
    if not od:
        reflected_eye = cv2.flip(reflected_eye, 1)
        sig = cv2.flip(sig, 1)
        str_side = 'os'

    cv2.imwrite('log/' + str(int(time.time())) + '.jpg', reflected_eye)
    cv2.imwrite('log/propSupport' + str_side + '.png', sig)
    with open('log/proptosis.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['{}'.format('od' if od else 'os')] + [prop])

    return prop, reflected_eye, sig


if __name__ == '__main__':
    '''od'''
    img_name = 'check/widthOrProp_od.jpg'
    frame = cv2.imread(img_name)
    prop_value, prop_frame, sig = proptosis(frame, offset=18.5, od=True)
    print(prop_value)
    # '''os'''
    # img_name = 'os-prop-patient1.jpg'
    # frame = cv2.imread(img_name)
    # prop_value, prop_frame, sig = proptosis(frame, offset=14.5, od=False)
    # print(prop_value)

    # for i in Path.cwd().rglob('prop/*.jpg'):
    #     print(i.name)
    #     frame = cv2.imread(str(i))
    #     prop_value, prop_frame = proptosis(frame, offset=12, od=False)
    #     time.sleep(1)
    #     print(prop_value)

    # import sys
    #
    # str_direction = sys.argv[1]  # 23.46-8.46L                #22.86-7.86R
    # str_pos = sys.argv[2]
    # str_pathpre = sys.argv[3]
    # str_path = sys.argv[4]
    # try:
    #     if str_direction == "left":
    #         left = cv2.imread(str_pathpre)
    #         res_l, frame_O, sig = proptosis(left, float(str_pos), od=False)
    #         cv2.imwrite('{}'.format(str_path), frame_O)
    #         width_frame, width_value = width(left, float(str_pos), od=False)
    #         width_path = str_path.split(".")[0] + "width.jpg"
    #         cv2.imwrite(width_path, width_frame)
    #         print("left")
    #         print("%.1f" % res_l)
    #         print(str_path)
    #         print("%.1f" % width_value)
    #         print(width_path)
    #     else:
    #         right = cv2.imread(str_pathpre)
    #         res_r, frame_O, sig = proptosis(right, float(str_pos), od=True)
    #         cv2.imwrite(str_path, frame_O)
    #         width_frame, width_value = width(right, float(str_pos), od=True)
    #         width_path = str_path.split(".")[0] + "width.jpg"
    #         cv2.imwrite(width_path, width_frame)
    #         print("right")
    #         print("%.1f" % res_r)
    #         print(str_path)
    #         print("%.1f" % width_value)
    #         print(width_path)
    # except Exception:
    #     print(str_direction + "error")
    #     print(str_direction + "error")
    #     print(str_direction + "error")
    #     print(str_direction + "error")
    #     print(str_direction + "error")
