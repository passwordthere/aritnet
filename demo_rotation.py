import csv
import io
import math
import os

from skimage.draw import line

from demo_play import segmentation, logger
from demo_utils import *

flow_w = 1536
flow_h = 1024
ratio = flow_h / 1024


def draw_plot(angles, filename):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    angles = angles + [angles[0]]
    ax.plot(np.array([x * (math.pi / 4) for x in range(9)]), np.array(angles), color='red', marker='None',
            linestyle='-', linewidth=3)
    plt.fill_between(np.array([x * (math.pi / 4) for x in range(9)]), np.array(angles), np.zeros(9), color='red',
                     alpha=0.3)
    ax.set_rmax(60)
    ax.set_rticks([x / 1.0 for x in range(0, 60, 10)], minor=False)
    ax.set_theta_zero_location("N")
    ax.set_rlabel_position(0)
    ax.grid(True)
    fig.savefig(filename, dpi=300)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
    buffer.close()
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    # plt.close()

    return image


def get_extended_corner(center1, center2, height, width):
    difference_x, difference_y = center1[0] - center2[0], center1[1] - center2[1]
    center1 = list(center1)
    center2 = list(center2)
    while 0 <= center2[0] < height and 0 <= center2[1] < width:
        center2[0] = center2[0] - difference_x
        center2[1] = center2[1] - difference_y
    while 0 <= center1[0] < height and 0 <= center1[1] < width:
        center1[0] = center1[0] + difference_x
        center1[1] = center1[1] + difference_y
    center1[0] = int(center1[0])
    center1[1] = int(center1[1])
    center2[0] = int(center2[0])
    center2[1] = int(center2[1])
    return center1, center2


def find_border_points(discrete_line, sig):
    point1 = [-1, -1]
    point2 = [-1, -1]

    sig = cv2.dilate(sig, np.ones((3, 1), np.uint8), iterations=1)

    for x, y in discrete_line:
        if 0 <= x < sig.shape[0] and 0 <= y < sig.shape[1]:
            if sig[:, :, 2][x, y] == 255:
                point1 = [x, y]
            if sig[:, :, 1][x, y] == 255:
                point2 = [x, y]
    return point1, point2


def get_orthogonal(k):
    k = k.astype('float32')
    k /= np.linalg.norm(k)
    x = np.random.randn(2)
    x -= x.dot(k) * k
    x /= np.linalg.norm(x)
    return x * 50


def find_tip_points(discrete_line, sig):
    point1 = [-1, -1]
    point2 = [-1, -1]

    sig = cv2.dilate(sig, np.ones((3, 1), np.uint8), iterations=1)

    point1s = []
    for x, y in discrete_line:
        if 0 <= x < sig.shape[0] and 0 <= y < sig.shape[1]:
            if sig[x, y] == 255:
                point1 = [x, y]
                point1s.append(point1)

    if point1s:
        point1 = point1s[0]
        point2 = point1s[-1]
    return point1, point2


def get_orthogonal_scale_difference(center1, center2, mask1, mask2, sig):
    center1 = np.array(center1)
    center2 = np.array(center2)
    vector = center1 - center2
    orthogonal = get_orthogonal(vector)
    point1 = center1 + orthogonal
    point2 = center2 + orthogonal
    start1, end1 = get_extended_corner(center1, point1, mask1.shape[0], mask1.shape[1])
    start2, end2 = get_extended_corner(center2, point2, mask2.shape[0], mask2.shape[1])
    discrete_line1 = list(zip(*line(*start1, *end1)))
    head1, tail1 = find_tip_points(discrete_line1, sig[:, :, 2])
    discrete_line2 = list(zip(*line(*start2, *end2)))
    head2, tail2 = find_tip_points(discrete_line2, sig[:, :, 1])
    length1 = math.sqrt(pow(head1[0] - tail1[0], 2) + pow(head1[1] - tail1[1], 2))
    length2 = math.sqrt(pow(head2[0] - tail2[0], 2) + pow(head2[1] - tail2[1], 2))
    if length2 == 0:
        return 1
    return length1 / length2


def get_angle(point1, point2, factor, radius):
    distance = math.sqrt(pow(point1[1] - point2[1], 2) + pow(point1[0] - point2[0], 2))
    if distance <= 1:
        angle = 0
    elif abs(distance * factor / radius) > 1.0:
        angle = 0
    else:
        angle = math.asin(distance * factor / radius)
    return angle * 180 / math.pi


def get_version_angle(mask1, pupil_ellipse1, mask2, pupil_ellipse2, mmPerPixel, sig):
    center1 = [round(pupil_ellipse1[1]), round(pupil_ellipse1[0])]
    center2 = [round(pupil_ellipse2[1]), round(pupil_ellipse2[0])]
    if center1[0] == center2[0] and center1[1] == center2[1]:
        center2[1] = center2[1] - 1
    extended1, extended2 = get_extended_corner(center1, center2, mask1.shape[0], mask1.shape[1])
    discrete_line = list(zip(*line(*extended2, *extended1)))
    point1, point2 = find_border_points(discrete_line, sig)
    if point2[0] == center1[0] and point2[1] == center1[1]:
        point2[1] = point2[1] - 1
    ratio = get_orthogonal_scale_difference(center1, center2, mask1, mask2, sig)
    # ratio = 1
    if (point1[0] <= center1[0] <= point2[0] and point1[1] <= center1[1] <= point2[1]) \
            or (point1[0] >= center1[0] >= point2[0] and point1[1] <= center1[1] <= point2[1]) \
            or (point1[0] <= center1[0] <= point2[0] and point1[1] >= center1[1] >= point2[1]) \
            or (point1[0] >= center1[0] >= point2[0] and point1[1] >= center1[1] >= point2[1]):
        point2[0] = int(center2[0] + ratio * (point2[0] - center2[0]))
        point2[1] = int(center2[1] + ratio * (point2[1] - center2[1]))
        angle1 = get_angle(point1, center1, mmPerPixel, 12.5)
        angle2 = get_angle(point2, center1, mmPerPixel, 12.5)
        angle = abs(angle1 + angle2)
    else:
        point2[0] = int(center2[0] + ratio * (point2[0] - center2[0]))
        point2[1] = int(center2[1] + ratio * (point2[1] - center2[1]))
        angle1 = get_angle(point1, center1, mmPerPixel, 12.5)
        angle2 = get_angle(point2, center1, mmPerPixel, 12.5)
        angle = abs(angle1 - angle2)

    return angle, extended1, point1, extended2, point2


def rotation(img_frames, od):
    if not od:
        img_frames = [cv2.flip(img_frame, 1) for img_frame in img_frames]

    seg_map0 = segmentation(img_frames[0])
    seg_map0 = cv2.resize(seg_map0, (1536, 1024), interpolation=cv2.INTER_NEAREST)

    pupilPts0, irisPts0 = getValidPoints(seg_map0)
    iris_ellipse0 = get_ellipse(pupilPts0, seg_map0)

    if iris_ellipse0 is None:
        logger.write_silent('rotation: ellipse[0] cannot be fitted properly')
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    img_gray = cv2.cvtColor(img_frames[0], cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    plot_segmap_ellpreds(img_gray, iris_ellipse0)
    cv2.circle(img_gray, (round(iris_ellipse0[0]), round(iris_ellipse0[1])), radius=9, color=(0, 0, 255), thickness=-1)

    pixelPerMm = pixelPerMm_R if od else pixelPerMm_L

    angles = []
    for i in range(1, 9, 1):
        seg_map = segmentation(img_frames[i])
        seg_map = cv2.resize(seg_map, (1536, 1024), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(str(int(time.time())) + '.jpg', seg_map * 255 / np.max(seg_map))

        pupilPts, irisPts = getValidPoints(seg_map)
        iris_ellipse = get_ellipse(pupilPts, seg_map)

        if iris_ellipse is None:
            logger.write_silent('rotation: ellipse[{}] cannot be fitted properly'.format(str(i)))
            angles.append(0.0)
            continue

        # print(iris_ellipse0, '::' , iris_ellipse)
        # rr0, cc0 = np.where(seg_map0 == 3)
        # rr1, cc1 = np.where(seg_map == 3)
        # if abs(rr0.min() - rr1.min()) < 2
        
        if abs(iris_ellipse0[0] - iris_ellipse[0]) < 5 and abs(iris_ellipse0[1] - iris_ellipse[1]) < 5:
            logger.write_silent('rotation: ellipse[{}] did not move'.format(str(i)))
            angles.append(0.0)
            continue

        sig = np.zeros(img_gray.shape).astype('uint8')
        plot_segmap_ellpreds(sig, iris_ellipse0, np.array([0, 0, 255]))
        plot_segmap_ellpreds(sig, iris_ellipse, np.array([0, 255, 0]))

        angle, extended0, point0, extended1, point1 = get_version_angle(seg_map0, iris_ellipse0, seg_map, iris_ellipse,
                                                                        1 / pixelPerMm, sig)
        cv2.circle(sig, (round(iris_ellipse0[0]), round(iris_ellipse0[1])), radius=9, color=(0, 0, 255), thickness=-1)
        cv2.circle(sig, (round(iris_ellipse[0]), round(iris_ellipse[1])), radius=9, color=(0, 0, 255), thickness=-1)
        cv2.circle(sig, (point0[1], point0[0]), radius=3, color=(255, 0, 0), thickness=-1)
        cv2.circle(sig, (point1[1], point1[0]), radius=3, color=(255, 0, 0), thickness=-1)
        os.makedirs('abc', exist_ok=True)
        cv2.imwrite(f'abc/{str(i)}.jpg', sig)

        plot_segmap_ellpreds(img_gray, iris_ellipse, np.array([128, 128, 128]))
        cv2.line(img_gray, (extended0[1], extended0[0]), (extended1[1], extended1[0]), (0, 255, 0), 1)
        cv2.circle(img_gray, (round(iris_ellipse[0]), round(iris_ellipse[1])), radius=9, color=(0, 0, 255), thickness=-1)
        cv2.circle(img_gray, (point0[1], point0[0]), radius=3, color=(255, 0, 0), thickness=-1)
        cv2.circle(img_gray, (point1[1], point1[0]), radius=3, color=(255, 0, 0), thickness=-1)

        angles.append(round(angle, 1))

    if not od:
        img_gray = cv2.flip(img_gray, 1)

    cv2.imwrite('log/' + str(int(time.time())) + '.jpg', img_gray)
    with open('log/rotation.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['{}'.format('od' if od else 'os')] + angles)

    return angles


def main(frames, od):
    angles = rotation(frames, od)
    timestamp = str(int(time.time()))
    if od:
        plot = draw_plot(angles, 'log/plot_right' + timestamp + '.png')
        # draw_plot(angles, path + 'plot_right.png')
    else:
        plot = draw_plot(angles, 'log/plot_left' + timestamp + '.png')
        # draw_plot(angles, path + 'plot_left.png')

    return angles, plot


if __name__ == '__main__':
    '''last generation ill'''
    frames = []
    for i in [5, 2, 1, 4, 7, 8, 9, 6, 3]:
        img_name = f'check/rotation/patient1_{i}_od.jpg'
        frame = cv2.imread(img_name)
        frames.append(frame)
    angles, plot = main(frames, od=True)
    print(angles)

    frames = []
    for i in [5, 2, 1, 4, 7, 8, 9, 6, 3]:
        img_name = f'check/rotation/patient1_{i}_os.jpg'
        frame = cv2.imread(img_name)
        frames.append(frame)
    angles, plot = main(frames, od=False)
    print(angles)

    '''8'''
    # for _ in range(5):
    #     frames = []
    #     for i in range(9):
    #         img_name = f'0/rotation_od{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=True)
    #
    #     frames = []
    #     for i in range(9):
    #         img_name = f'0/rotation_os{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=False)

    """
    标定物
    """
    # for _ in range(5):
    #     '''od'''
    #     frames = []
    #     for i in range(9):
    #         img_name = f'sta/sta15/OD_{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=True)
    #     '''os'''
    #     frames = []
    #     for i in range(9):
    #         img_name = f'sta/sta15/OS_{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=False)
    #     '''od'''
    #     frames = []
    #     for i in range(9):
    #         img_name = f'sta/sta30/OD_{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=True)
    #     '''os'''
    #     frames = []
    #     for i in range(9):
    #         img_name = f'sta/sta30/OS_{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=False)
    #     '''od'''
    #     frames = []
    #     for i in range(9):
    #         img_name = f'sta/sta45/OD_{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=True)
    #     '''os'''
    #     frames = []
    #     for i in range(9):
    #         img_name = f'sta/sta45/OS_{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=False)
    #     '''od'''
    #     frames = []
    #     for i in range(9):
    #         img_name = f'sta/sta61/OD_{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=True)
    #     '''os'''
    #     frames = []
    #     for i in range(9):
    #         img_name = f'sta/sta61/OS_{i}.bmp'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=False)

    """
    8号机（小朱）
    """
    # for _ in range(5):
    #     frames = []
    #     for i in [5, 2, 3, 6, 9, 8, 7, 4, 1]:
    #         img_name = f'小朱379811989808771563/right_{i}.jpg'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=True)
    #
    #
    #     frames = []
    #     for i in [5, 2, 3, 6, 9, 8, 7, 4, 1]:
    #         img_name = f'小朱379811989808771563/left_{i}.jpg'
    #         frame = cv2.imread(img_name)
    #         frames.append(frame)
    #     angles = main(frames, 'log/', od=False)

#     import sys
#
#     lookup = {'0': 2,
#               '3': 3,
#               '6': 4,
#               '1': 1,
#               '4': 0,
#               '7': 5,
#               '2': 8,
#               '5': 7,
#               '8': 6}
#     result = {
#         'rotation_left': ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
#         'rotation_right': ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
#     }
#     images_list_left = [None, None, None, None, None, None, None, None, None]
#     images_list_right = [None, None, None, None, None, None, None, None, None]
#     str_pos_left = sys.argv[1]
#     str_pos_right = sys.argv[2]
#     str_path = sys.argv[3]
# #     '''os'''
# try:
#     for i in range(9):
#         images_list_left[lookup[str(i)]] = cv2.imread((str_path + "left_{}.jpg").format(i + 1), cv2.IMREAD_COLOR)
#
#     result['rotation_left'] = main(images_list_left, str_path, od=False)
#     str_l = "1eft"
#     result_l = result['rotation_left']
#     pic_l = str_path + "plot_left.png"
# except Exception:
#     str_l = "lefterror"
#     result_l = "lefterror"
#     pic_l = "lefterror"
#
# try:
#     for i in range(9):
#         images_list_right[lookup[str(i)]] = cv2.imread((str_path + "right_{}.jpg").format(i + 1), cv2.IMREAD_COLOR)
#
#     result['rotation_right'] = main(images_list_right, str_path, od=True)
#     str_r = "right"
#     result_r = result['rotation_right']
#     pic_r = str_path + "plot_right.png"
# except Exception:
#     str_r = "righterror"
#     result_r = "righterror"
#     pic_r = "righterror"
#
# print(str_l)
# print(result_l)
# print(pic_l)
# print(str_r)
# print(result_r)
# print(pic_r)
